"""
Ablation experiment for TuBERT (section 5.3.1).

Four conditions:
  full            — audio + text (normal TuBERT)
  text_only       — audio tensor zeroed out
  audio_only      — text tensor zeroed out
  distilbert_only — raw DistilBERT classifier, no RNN

Usage:
    python ablation.py
    python ablation.py --dataset iemocap
    python ablation.py --mode text_only audio_only
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import softmax
from torchmetrics import Accuracy, F1Score, Precision, Recall
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from model.setup.data_loader import IEMOCAPDataset, MELDDataset
from model.setup.feature_extractor import AudioFeatureExtractor, TextFeatureExtractor
from predictor_classes import EmotionPredictor

# ── DistilBERT baseline constants ─────────────────────────────────────────────

# The fine-tuned DistilBERT model outputs these 6 labels (in this order).
DISTILBERT_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Neutral confidence threshold: if top prediction < this, return neutral.
DISTILBERT_NEUTRAL_THRESHOLD = 0.50

# Map DistilBERT labels → TuBERT emotion set.
# "love" has no direct equivalent; treat as joy (closest positive emotion).
# "fear" is merged into anger (matching the MELD class merge used in training).
DISTILBERT_TO_TUBERT = {
    "sadness": "sadness",
    "joy": "joy",
    "love": "joy",
    "anger": "anger",
    "fear": "anger",
    "surprise": "surprise",
}

# ── IEMOCAP helpers ───────────────────────────────────────────────────────────

IEMOCAP_TO_MODEL_EMOTION = {
    "neutral": "neutral",
    "happiness": "joy",
    "sadness": "sadness",
    "anger": "anger",
}
IEMOCAP_SURPRISE_REMAP = "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# DistilBERT-only baseline
# ─────────────────────────────────────────────────────────────────────────────


class DistilBERTBaseline:
    """
    Uses the fine-tuned DistilBERT sequence classifier directly.
    No audio, no RNN — pure text → emotion.
    """

    def __init__(self, device=config.DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.DISTILBERT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.DISTILBERT_MODEL
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        """
        Returns:
            emotion (str): predicted TuBERT-compatible emotion
            prob_dict (dict): {tubert_emotion: probability} after merging
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = softmax(logits, dim=1).squeeze(0).cpu().numpy()

        raw = {label: float(probs[i]) for i, label in enumerate(DISTILBERT_LABELS)}

        # Merge into TuBERT emotion set (fear → anger, love → joy)
        merged = {e: 0.0 for e in config.EMOTIONS}
        for db_label, prob in raw.items():
            tubert_label = DISTILBERT_TO_TUBERT[db_label]
            if tubert_label in merged:
                merged[tubert_label] += prob

        # DistilBERT has no neutral class — apply confidence threshold
        top_emotion = max(merged, key=merged.get)
        if merged[top_emotion] < DISTILBERT_NEUTRAL_THRESHOLD:
            top_emotion = "neutral"

        return top_emotion, merged


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_metrics(num_classes):
    return dict(
        accuracy=Accuracy(task="multiclass", num_classes=num_classes),
        precision_macro=Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ),
        precision_weighted=Precision(
            task="multiclass", num_classes=num_classes, average="weighted"
        ),
        precision_per=Precision(
            task="multiclass", num_classes=num_classes, average=None
        ),
        recall_macro=Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ),
        recall_weighted=Recall(
            task="multiclass", num_classes=num_classes, average="weighted"
        ),
        recall_per=Recall(task="multiclass", num_classes=num_classes, average=None),
        f1_macro=F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        f1_weighted=F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        ),
        f1_per=F1Score(task="multiclass", num_classes=num_classes, average=None),
    )


def _get_zero_audio_features():
    """Zeroed audio feature array shaped (1, AUDIO_FEATURE_DIM)."""
    return np.zeros((1, config.AUDIO_FEATURE_DIM), dtype=np.float32)


def _get_zero_text_embedding():
    """Zeroed text embedding array shaped (TEXT_EMBEDDING_DIM,)."""
    return np.zeros(config.TEXT_EMBEDDING_DIM, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MELD ablation
# ─────────────────────────────────────────────────────────────────────────────


def run_ablation_meld(modes, output_lines, path):
    test_dataset = MELDDataset(split="test")
    audio_extractor = AudioFeatureExtractor()
    text_extractor = TextFeatureExtractor(device=config.DEVICE)

    tubert_needed = any(m in modes for m in ("full", "text_only", "audio_only"))
    tubert = EmotionPredictor(model_path=path) if tubert_needed else None
    distilbert = DistilBERTBaseline() if "distilbert_only" in modes else None

    num_classes = len(config.EMOTIONS)
    results = {mode: {"preds": [], "actual": []} for mode in modes}

    for i in tqdm(range(len(test_dataset)), desc="MELD ablation"):
        audio_path, metadata = test_dataset[i]
        text = metadata["transcript"]

        emotion = metadata["emotion"]
        if emotion == "disgust":
            emotion = "anger"
        if emotion == "fear":
            emotion = "sadness"
        if emotion not in config.EMOTION_TO_IDX:
            continue
        label = config.EMOTION_TO_IDX[emotion]

        if tubert_needed and Path(audio_path).is_file():
            audio_feats = audio_extractor.extract(audio_path)
            text_emb = text_extractor.extract(text)
        else:
            audio_feats = None
            text_emb = None

        for mode in modes:
            if mode == "distilbert_only":
                pred_str, _ = distilbert.predict(text)
                pred_idx = config.EMOTION_TO_IDX.get(
                    pred_str, config.EMOTION_TO_IDX["neutral"]
                )
            else:
                if audio_feats is None:
                    continue
                if mode == "full":
                    af, te = audio_feats, text_emb
                elif mode == "text_only":
                    af, te = _get_zero_audio_features(), text_emb
                elif mode == "audio_only":
                    af, te = audio_feats, _get_zero_text_embedding()

                pred_str, _, _ = tubert.predict_from_features(af, te)
                pred_idx = config.EMOTION_TO_IDX[pred_str]

            results[mode]["preds"].append(pred_idx)
            results[mode]["actual"].append(label)

    _print_ablation_results(
        results,
        modes,
        num_classes,
        config.EMOTIONS,
        "MELD TEST SET — ABLATION",
        output_lines,
    )


# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP ablation
# ─────────────────────────────────────────────────────────────────────────────


def run_ablation_iemocap(modes, output_lines, path, eval_class=5):
    test_dataset = IEMOCAPDataset()
    audio_extractor = AudioFeatureExtractor()
    text_extractor = TextFeatureExtractor(device=config.DEVICE)

    tubert_needed = any(m in modes for m in ("full", "text_only", "audio_only"))
    tubert = EmotionPredictor(model_path=path) if tubert_needed else None
    distilbert = DistilBERTBaseline() if "distilbert_only" in modes else None

    iemocap_emotions = config.EMOTIONS_IEMOCAP
    iemocap_to_idx = {e: i for i, e in enumerate(iemocap_emotions)}
    num_classes = len(iemocap_emotions)
    results = {mode: {"preds": [], "actual": []} for mode in modes}
    skipped = 0

    for i in tqdm(range(len(test_dataset)), desc="IEMOCAP ablation"):
        audio_path, metadata = test_dataset[i]
        audio_path = str(audio_path)
        consensus = metadata["consensus_emotion"]
        session = metadata["session_number"]

        if consensus not in IEMOCAP_TO_MODEL_EMOTION or session != eval_class:
            skipped += 1
            continue

        emotion = IEMOCAP_TO_MODEL_EMOTION[consensus]
        text = metadata["transcription"]
        label = iemocap_to_idx[emotion]

        if tubert_needed and Path(audio_path).is_file():
            audio_feats = audio_extractor.extract(audio_path)
            text_emb = text_extractor.extract(text)
        else:
            audio_feats = None
            text_emb = None

        for mode in modes:
            if mode == "distilbert_only":
                pred_str, _ = distilbert.predict(text)
                if pred_str == "surprise":
                    pred_str = IEMOCAP_SURPRISE_REMAP
                pred_idx = iemocap_to_idx.get(pred_str, iemocap_to_idx["neutral"])
            else:
                if audio_feats is None:
                    continue
                if mode == "full":
                    af, te = audio_feats, text_emb
                elif mode == "text_only":
                    af, te = _get_zero_audio_features(), text_emb
                elif mode == "audio_only":
                    af, te = audio_feats, _get_zero_text_embedding()

                pred_str, _, _ = tubert.predict_from_features(af, te)
                if pred_str == "surprise":
                    pred_str = IEMOCAP_SURPRISE_REMAP
                pred_idx = iemocap_to_idx.get(pred_str, iemocap_to_idx["neutral"])

            results[mode]["preds"].append(pred_idx)
            results[mode]["actual"].append(label)

    if skipped:
        msg = f"\nSkipped {skipped} IEMOCAP samples with unmappable 'other' label."
        print(msg)
        output_lines.append(msg)

    _print_ablation_results(
        results,
        modes,
        num_classes,
        iemocap_emotions,
        "IEMOCAP — ABLATION",
        output_lines,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printing
# ─────────────────────────────────────────────────────────────────────────────

MODE_LABELS = {
    "full": "Full TuBERT (audio + text)",
    "text_only": "TuBERT Text-Only Input    ",
    "audio_only": "TuBERT Audio-Only Input   ",
    "distilbert_only": "Base DistilBERT           ",
}


def _print_ablation_results(results, modes, num_classes, emotions, title, output_lines):
    def log(line=""):
        print(line)
        output_lines.append(line)

    log()
    log("=" * 70)
    log(title)
    log("=" * 70)

    log(
        f"\n{'Condition':<36} {'Acc':>6}  {'P (mac)':>8}  {'R (mac)':>8}  {'F1 (mac)':>9}  {'F1 (wtd)':>9}"
    )
    log("-" * 84)

    per_class_blocks = []

    for mode in modes:
        preds = results[mode]["preds"]
        actual = results[mode]["actual"]
        if not preds:
            log(f"{MODE_LABELS[mode]}  — no predictions (check file paths)")
            continue

        t_preds = torch.tensor(preds, dtype=torch.long)
        t_actual = torch.tensor(actual, dtype=torch.long)

        m = _make_metrics(num_classes)
        for metric in m.values():
            metric.update(t_preds, t_actual)

        acc = m["accuracy"].compute().item()
        p_mac = m["precision_macro"].compute().item()
        r_mac = m["recall_macro"].compute().item()
        f1_mac = m["f1_macro"].compute().item()
        f1_wtd = m["f1_weighted"].compute().item()
        f1_per = m["f1_per"].compute()

        log(
            f"{MODE_LABELS[mode]}  {acc:>6.2%}  {p_mac:>8.2%}  {r_mac:>8.2%}  {f1_mac:>9.2%}  {f1_wtd:>9.2%}"
        )
        per_class_blocks.append((mode, f1_per))

    log()
    log("─" * 70)
    log("PER-CLASS F1 SCORES")
    log("─" * 70)
    log(f"{'Emotion':<12}" + "".join(f"{MODE_LABELS[m]:>22}" for m in modes))
    log("-" * (12 + 22 * len(modes)))

    for ei, emo in enumerate(emotions):
        row = f"{emo:<12}"
        for _, f1_per in per_class_blocks:
            val = f1_per[ei].item() if ei < len(f1_per) else float("nan")
            row += f"{val:>22.2%}"
        log(row)

    log()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODES = ["full", "text_only", "audio_only", "distilbert_only"]
ALL_DATASETS = ["meld", "iemocap"]


def main():
    parser = argparse.ArgumentParser(
        description="TuBERT ablation experiment (section 5.3.1)"
    )
    parser.add_argument(
        "--dataset",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        help="Dataset to evaluate on (default: meld)",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=ALL_MODES,
        default=ALL_MODES,
        help="Which conditions to run (default: all four)",
    )
    args = parser.parse_args()

    output_lines = []
    model_path = config.MODELS_DIR / "iemocap_best.pt"

    if "meld" in args.dataset:
        run_ablation_meld(args.mode, output_lines, model_path)
    if "iemocap" in args.dataset:
        run_ablation_iemocap(args.mode, output_lines, model_path)

    config.STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.STATS_DIR / f"{model_path.stem}_ablation_results.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
