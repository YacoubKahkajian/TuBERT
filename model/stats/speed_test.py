"""

Measures per-utterance latency across four pipeline phases on the MELD test
split, following the methodology in Chamishka et al.

Phases timed:
    1. Audio Feature Extraction: MFCC + deltas + CMVN via AudioFeatureExtractor
    2. Vosk Transcription: KaldiRecognizer speech-to-text
    3. Text Embedding: DistilBERT [CLS] token embedding
    4. Emotion Prediction: MultiModalEmotionRNN forward pass

Usage:
    python benchmark_speed.py
    python benchmark_speed.py --output file
    python benchmark_speed.py --output both --threshold 1.0
    python benchmark_speed.py --warmup 10

Run from the project root (where config.py lives).
"""

import argparse
import json
import logging
import statistics
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from vosk import KaldiRecognizer
from vosk import Model as VoskModel

import config
from model.setup.data_loader import MELDDataset
from model.setup.feature_extractor import AudioFeatureExtractor, TextFeatureExtractor
from model.setup.rnn import MultiModalEmotionRNN

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)
logger.propagate = False


def setup_logging(output: str, log_path: Path) -> None:
    formatter = logging.Formatter("%(message)s")
    if output in ("terminal", "both"):
        logger.addHandler(logging.StreamHandler(sys.stdout))
    if output in ("file", "both"):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    for h in logger.handlers:
        h.setFormatter(formatter)


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------


class Timer:
    """Accumulates per-utterance timings for a single pipeline phase."""

    def __init__(self, name: str):
        self.name = name
        self.times_ms: list[float] = []
        self._start: float = 0.0

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        elapsed = (time.perf_counter() - self._start) * 1000.0
        self.times_ms.append(elapsed)
        return elapsed

    # --- Aggregate stats ---

    @property
    def total_ms(self) -> float:
        return sum(self.times_ms)

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        return s[int(0.95 * len(s))]

    @property
    def p99_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        s = sorted(self.times_ms)
        return s[int(0.99 * len(s))]

    @property
    def latency_s(self) -> float:
        """Average per-utterance latency in seconds (the key RT metric)."""
        return self.avg_ms / 1000.0

    def avg_per_conversation(self, conv_count: int) -> float:
        return self.total_ms / conv_count if conv_count else 0.0

    @property
    def count(self) -> int:
        return len(self.times_ms)


# ---------------------------------------------------------------------------
# Model / pipeline loading (not timed)
# ---------------------------------------------------------------------------


def load_pipeline(model_path: str):
    """Load all models once before the benchmark loop."""
    print("Loading audio feature extractor...")
    audio_extractor = AudioFeatureExtractor()

    print("Loading DistilBERT text feature extractor (CPU)...")
    text_extractor = TextFeatureExtractor(device="cpu")

    print("Loading Vosk speech-to-text model...")
    vosk_model = VoskModel(lang="en-us")

    print(f"Loading TuBERT checkpoint from {model_path}...")
    model = MultiModalEmotionRNN()
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to("cpu")

    return audio_extractor, text_extractor, vosk_model, model


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def run_benchmark(
    audio_extractor: AudioFeatureExtractor,
    text_extractor: TextFeatureExtractor,
    vosk_model: VoskModel,
    model: MultiModalEmotionRNN,
    warmup: int = 5,
):
    """
    Run the benchmark on every utterance in the MELD test split.

    Returns:
        timers:         dict of phase_name -> Timer
        num_utterances: int
        num_convs:      int
        skipped:        int
    """
    dataset = MELDDataset(split="test")

    # --- Collect valid utterances & group by conversation ---
    utterances = []  # list of (audio_path, transcript, dialogue_id)
    for idx in range(len(dataset)):
        audio_path, metadata = dataset[idx]
        if Path(audio_path).is_file():
            utterances.append(
                (
                    audio_path,
                    metadata["transcript"],
                    metadata["dialogue_id"],
                )
            )
    skipped = len(dataset) - len(utterances)
    dialogue_ids = set(u[2] for u in utterances)
    num_convs = len(dialogue_ids)

    print(
        f"  MELD test split: {len(dataset)} total, "
        f"{len(utterances)} with audio, {skipped} skipped"
    )
    print(f"  Conversations: {num_convs}")
    print(f"  Avg utterances/conversation: {len(utterances) / num_convs:.1f}")

    # --- Warmup (not timed) ---
    print(f"  Warming up ({warmup} utterances)...", flush=True)
    for i in range(min(warmup, len(utterances))):
        path, text, _ = utterances[i]
        audio_feats = audio_extractor.extract(path)
        text_emb = text_extractor.extract(text)
        audio_tensor = torch.FloatTensor(audio_feats).unsqueeze(0)
        text_tensor = torch.FloatTensor(text_emb).unsqueeze(0)
        length_tensor = torch.LongTensor([audio_feats.shape[0]])
        with torch.no_grad():
            model(audio_tensor, text_tensor, length_tensor)

    # --- Timed benchmark ---
    t_audio = Timer("Audio Feature Extraction")
    t_vosk = Timer("Vosk Transcription")
    t_text = Timer("Text Embedding")
    t_pred = Timer("Emotion Prediction")

    for audio_path, transcript, dia_id in tqdm(utterances, desc="Benchmarking"):
        # Phase 1: Audio feature extraction (MFCC + deltas + CMVN)
        t_audio.start()
        audio_features = audio_extractor.extract(audio_path)
        t_audio.stop()

        # Phase 2: Vosk transcription
        t_vosk.start()
        wf = wave.open(audio_path, "rb")
        rec = KaldiRecognizer(vosk_model, 16000)
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)
        vosk_result = json.loads(rec.FinalResult())
        vosk_text = vosk_result.get("text", "")
        wf.close()
        t_vosk.stop()

        # Phase 3: Text embedding (DistilBERT)
        # Use MELD ground truth transcript so embedding quality matches
        # the accuracy results in section 5.2.1; Vosk is timed separately.
        t_text.start()
        text_embedding = text_extractor.extract(transcript)
        t_text.stop()

        # Phase 4: Emotion prediction (RNN forward pass only)
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0)
        text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0)
        length_tensor = torch.LongTensor([audio_features.shape[0]])

        t_pred.start()
        with torch.no_grad():
            logits = model(audio_tensor, text_tensor, length_tensor)
        t_pred.stop()

    timers = {t.name: t for t in [t_audio, t_vosk, t_text, t_pred]}
    return timers, len(utterances), num_convs, skipped


# ---------------------------------------------------------------------------
# Reporting (mirrors Table 7)
# ---------------------------------------------------------------------------


def print_report(
    timers: dict[str, Timer],
    num_utterances: int,
    num_convs: int,
    skipped: int,
    threshold_s: float,
):
    phases = list(timers.values())
    total_latency_s = sum(t.latency_s for t in phases)

    logger.info("")
    logger.info("=" * 105)
    logger.info("RESULTS — Real-time Performance Statistics")
    logger.info("(cf. Table 7, Chamishka et al. 2022)")
    logger.info("=" * 105)

    header = (
        f"{'Phase':<28}"
        f"{'Total (ms)':>14}"
        f"{'Avg/Utt (ms)':>14}"
        f"{'Avg/Conv (ms)':>16}"
        f"{'RT Latency (s)':>16}"
    )
    logger.info(header)
    logger.info("-" * 105)

    for t in phases:
        logger.info(
            f"{t.name:<28}"
            f"{t.total_ms:>14,.2f}"
            f"{t.avg_ms:>14.3f}"
            f"{t.avg_per_conversation(num_convs):>16.3f}"
            f"{t.latency_s:>16.3f}"
        )

    logger.info("-" * 105)
    logger.info(
        f"{'TOTAL':<28}"
        f"{sum(t.total_ms for t in phases):>14,.2f}"
        f"{sum(t.avg_ms for t in phases):>14.3f}"
        f"{sum(t.avg_per_conversation(num_convs) for t in phases):>16.3f}"
        f"{total_latency_s:>16.3f}"
    )

    # --- Extended stats ---
    logger.info("")
    logger.info("=" * 105)
    logger.info("EXTENDED STATISTICS — Per-Utterance Distribution")
    logger.info("=" * 105)

    header2 = (
        f"{'Phase':<28}"
        f"{'Std (ms)':>12}"
        f"{'Median (ms)':>14}"
        f"{'P95 (ms)':>12}"
        f"{'P99 (ms)':>12}"
        f"{'N':>8}"
    )
    logger.info(header2)
    logger.info("-" * 105)

    for t in phases:
        logger.info(
            f"{t.name:<28}"
            f"{t.std_ms:>12.3f}"
            f"{t.median_ms:>14.3f}"
            f"{t.p95_ms:>12.3f}"
            f"{t.p99_ms:>12.3f}"
            f"{t.count:>8}"
        )

    # --- Real-time feasibility ---
    logger.info("")
    logger.info("=" * 105)
    logger.info("REAL-TIME FEASIBILITY")
    logger.info("=" * 105)
    logger.info(f"  Total pipeline latency per utterance: {total_latency_s:.3f}s")
    logger.info(f"  Real-time threshold:                  {threshold_s:.3f}s")

    if total_latency_s <= threshold_s:
        logger.info("  Status: PASS")
    else:
        overshoot = total_latency_s - threshold_s
        logger.info(f"  Status: FAIL (exceeds threshold by {overshoot:.3f}s)")
        bottleneck = max(phases, key=lambda t: t.latency_s)
        pct = bottleneck.latency_s / total_latency_s * 100
        logger.info(
            f"  Bottleneck: {bottleneck.name} "
            f"({bottleneck.latency_s:.3f}s, {pct:.1f}% of total)"
        )

    logger.info("")
    logger.info(f"  Utterances benchmarked: {num_utterances}")
    logger.info(f"  Utterances skipped (no audio file): {skipped}")
    logger.info(f"  Conversations: {num_convs}")
    logger.info(f"  Device: CPU")
    logger.info("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="TuBERT speed benchmark (Section 5.2.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=config.MODEL_FILENAME,
        help="Model checkpoint filename (relative to config.MODELS_DIR)",
    )
    parser.add_argument(
        "--output",
        choices=["terminal", "file", "both"],
        default="terminal",
        help="Where to write results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Real-time latency threshold in seconds",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup utterances (not timed)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = config.MODELS_DIR / args.model
    log_path = config.STATS_DIR / f"{model_path.stem}_speed_results.txt"

    setup_logging(args.output, log_path)

    logger.info("=" * 105)
    logger.info("TuBERT SPEED BENCHMARK")
    logger.info("=" * 105)
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Dataset: MELD test split")
    logger.info(f"  Device: CPU")
    logger.info("")

    # Load everything (not timed)
    audio_ext, text_ext, vosk_model, rnn_model = load_pipeline(str(model_path))

    # Run benchmark
    timers, num_utts, num_convs, skipped = run_benchmark(
        audio_ext,
        text_ext,
        vosk_model,
        rnn_model,
        warmup=args.warmup,
    )

    # Report
    print_report(timers, num_utts, num_convs, skipped, args.threshold)

    if args.output_json:
        save_json(timers, num_utts, num_convs, Path(args.output_json))

    if args.output in ("file", "both"):
        logger.info(f"Results saved to {log_path}")


if __name__ == "__main__":
    main()
