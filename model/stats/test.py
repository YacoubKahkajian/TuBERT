"""
Evaluate the final model on the MELD test dataset or IEMOCAP dataset.

Usage:
    # Test both datasets, emotion mode, print to terminal (default)
    python test.py

    # Save results to a text file
    python test.py --output file

    # Test only MELD, sentiment mode
    python test.py --dataset meld --mode sentiment

    # Test only IEMOCAP, both modes, save to file
    python test.py --dataset iemocap --mode both --output file

    # Full custom run
    python test.py --dataset both --mode both --output both --model e_best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm

import config
from model.setup.data_loader import IEMOCAPDataset, MELDDataset
from predictor_classes import EmotionPredictor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("test")
logger.setLevel(logging.INFO)
# Prevent messages from propagating to the root logger
logger.propagate = False


def setup_logging(output: str, log_path: Path) -> None:
    """
    Configure the module-level logger based on the desired output destination.

    Args:
        output: One of "terminal", "file", or "both".
        log_path: Path to the log file (used when output is "file" or "both").
    """
    formatter = logging.Formatter("%(message)s")

    if output in ("terminal", "both"):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if output in ("file", "both"):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# IEMOCAP label mappings
# ---------------------------------------------------------------------------

# IEMOCAP's full label set mapped to this model's emotion set.
# "happiness" is called "joy" in this model; "other" has no mapping and is skipped.
IEMOCAP_TO_MODEL_EMOTION = {
    "neutral": "neutral",
    "happiness": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "surprise": "surprise",
}

# When use_four_emotions=True, surprise is excluded to match past literature that
# reports results on 4 classes (neutral, joy, sadness, anger) rather than 5.
# Both ground-truth "surprise" labels and "surprise" predictions are remapped.
IEMOCAP_SURPRISE_REMAP = "neutral"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _log_section(title: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# MELD evaluation
# ---------------------------------------------------------------------------


def test_meld(use_all_emotions: bool, model_path: Path) -> None:
    """
    Evaluate the final model on the MELD test dataset.

    Args:
        use_all_emotions: When True use all 7 emotion classes;
                          when False collapse to 3 sentiment classes.
        model_path: Path to the model checkpoint.
    """
    mode_label = "EMOTION" if use_all_emotions else "SENTIMENT"
    _log_section(f"MELD — {mode_label} RESULTS")

    model = EmotionPredictor(model_path=str(model_path))
    test_dataset = MELDDataset(split="test")

    num_classes = len(config.EMOTIONS) if use_all_emotions else len(config.SENTIMENTS)
    emotions = config.EMOTIONS if use_all_emotions else config.SENTIMENTS

    confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    precision_macro = Precision(
        task="multiclass", num_classes=num_classes, average="macro"
    )
    precision_per_class = Precision(
        task="multiclass", num_classes=num_classes, average=None
    )
    precision_weighted = Precision(
        task="multiclass", num_classes=num_classes, average="weighted"
    )

    recall_macro = Recall(task="multiclass", num_classes=num_classes, average="macro")
    recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
    recall_weighted = Recall(
        task="multiclass", num_classes=num_classes, average="weighted"
    )

    f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
    f1_weighted = F1Score(
        task="multiclass", num_classes=num_classes, average="weighted"
    )

    preds = []
    actual = []

    for i in tqdm(range(len(test_dataset)), desc=f"MELD {mode_label}"):
        audio_path, metadata = test_dataset[i]
        text = metadata["transcript"]
        emotion = metadata["emotion"]

        if emotion == "disgust":
            emotion = "anger"
        if emotion == "fear":
            emotion = "sadness"

        label = config.EMOTION_TO_IDX[emotion]
        predicted_label, _probs, _confidence = model.predict(audio_path, text)

        if not use_all_emotions:
            label = config.SENTIMENT_TO_IDX[config.EMOTION_TO_SENTIMENT[emotion]]
            predicted_label = config.SENTIMENT_TO_IDX[
                config.EMOTION_TO_SENTIMENT[predicted_label]
            ]
        else:
            predicted_label = config.EMOTION_TO_IDX[predicted_label]

        preds.append(predicted_label)
        actual.append(label)

    tensor_preds = torch.Tensor(preds)
    tensor_actual = torch.Tensor(actual)

    confusion_matrix.update(tensor_preds, tensor_actual)
    accuracy.update(tensor_preds, tensor_actual)
    precision_macro.update(tensor_preds, tensor_actual)
    precision_per_class.update(tensor_preds, tensor_actual)
    precision_weighted.update(tensor_preds, tensor_actual)
    recall_macro.update(tensor_preds, tensor_actual)
    recall_per_class.update(tensor_preds, tensor_actual)
    recall_weighted.update(tensor_preds, tensor_actual)
    f1_macro.update(tensor_preds, tensor_actual)
    f1_per_class.update(tensor_preds, tensor_actual)
    f1_weighted.update(tensor_preds, tensor_actual)

    # ── Overall metrics ──────────────────────────────────────────────────────
    _log_section("OVERALL METRICS")
    logger.info(f"Accuracy: {accuracy.compute():.4f}")
    logger.info("")
    logger.info("Macro Averages (unweighted):")
    logger.info(f"  Precision: {precision_macro.compute():.4f}")
    logger.info(f"  Recall:    {recall_macro.compute():.4f}")
    logger.info(f"  F1 Score:  {f1_macro.compute():.4f}")
    logger.info("")
    logger.info("Weighted Averages (by class frequency):")
    logger.info(f"  Precision: {precision_weighted.compute():.4f}")
    logger.info(f"  Recall:    {recall_weighted.compute():.4f}")
    logger.info(f"  F1 Score:  {f1_weighted.compute():.4f}")

    # ── Per-class metrics ────────────────────────────────────────────────────
    _log_section("PER-CLASS METRICS")
    precision_vals = precision_per_class.compute()
    recall_vals = recall_per_class.compute()
    f1_vals = f1_per_class.compute()

    logger.info(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    logger.info("-" * 60)
    for i, emotion in enumerate(emotions):
        logger.info(
            f"{emotion:<12} {precision_vals[i]:.4f}       {recall_vals[i]:.4f}       {f1_vals[i]:.4f}"
        )

    # ── Class distribution ───────────────────────────────────────────────────
    _log_section("CLASS DISTRIBUTION")
    unique, counts = torch.unique(tensor_actual, return_counts=True)
    for idx, count in zip(unique, counts):
        emotion = emotions[int(idx)]
        percentage = (count / len(tensor_actual)) * 100
        logger.info(f"{emotion:<12}: {count:>4} samples ({percentage:>5.1f}%)")

    # ── Confusion matrix plot ────────────────────────────────────────────────
    fig, _ax = confusion_matrix.plot(labels=emotions, cmap="Blues")
    filename = (
        f"{model_path.stem}_meld_emotion_confusion_matrix"
        if use_all_emotions
        else f"{model_path.stem}_meld_sentiment_confusion_matrix"
    )
    save_path = config.STATS_DIR / f"{filename}.pdf"
    fig.savefig(save_path, format="pdf")
    logger.info(f"\nConfusion matrix saved to {save_path}")


# ---------------------------------------------------------------------------
# IEMOCAP evaluation
# ---------------------------------------------------------------------------


def test_iemocap(
    use_all_emotions: bool,
    model_path: Path,
    use_four_emotions: bool = False,
) -> None:
    """
    Evaluate the final model on the IEMOCAP dataset.

    Uses the consensus_emotion field (majority-vote across annotators) as the
    ground-truth label.  Samples whose consensus label is "other" are skipped
    because they cannot be mapped to the model's emotion set.

    Args:
        use_all_emotions: When True use emotion classes; when False collapse to
                          3 sentiment classes.
        model_path: Path to the model checkpoint.
        use_four_emotions: When True, remap both ground-truth and predicted
                           "surprise" labels to IEMOCAP_SURPRISE_REMAP ("neutral"),
                           producing a 4-class evaluation (neutral, joy, sadness,
                           anger) comparable to past literature that excludes
                           surprise. Has no effect when use_all_emotions=False.
    """
    if use_four_emotions and use_all_emotions:
        mode_label = "EMOTION (4-CLASS, NO SURPRISE)"
    elif use_all_emotions:
        mode_label = "EMOTION (5-CLASS)"
    else:
        mode_label = "SENTIMENT"
    _log_section(f"IEMOCAP — {mode_label} RESULTS")

    model = EmotionPredictor(model_path=str(model_path))
    test_dataset = IEMOCAPDataset()

    if use_four_emotions and use_all_emotions:
        # Exclude surprise from the class list entirely
        iemocap_emotions = [e for e in config.EMOTIONS if e != "surprise"]
    else:
        iemocap_emotions = config.EMOTIONS
    iemocap_emotion_to_idx = {e: i for i, e in enumerate(iemocap_emotions)}

    num_classes = len(iemocap_emotions) if use_all_emotions else len(config.SENTIMENTS)
    emotions = iemocap_emotions if use_all_emotions else config.SENTIMENTS

    confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
    weighted_accuracy = Accuracy(
        task="multiclass", num_classes=num_classes, average="weighted"
    )

    precision_macro = Precision(
        task="multiclass", num_classes=num_classes, average="macro"
    )
    precision_per_class = Precision(
        task="multiclass", num_classes=num_classes, average=None
    )
    precision_weighted = Precision(
        task="multiclass", num_classes=num_classes, average="weighted"
    )

    recall_macro = Recall(task="multiclass", num_classes=num_classes, average="macro")
    recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
    recall_weighted = Recall(
        task="multiclass", num_classes=num_classes, average="weighted"
    )

    f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
    f1_weighted = F1Score(
        task="multiclass", num_classes=num_classes, average="weighted"
    )

    preds = []
    actual = []
    skipped = 0

    for i in tqdm(range(len(test_dataset)), desc=f"IEMOCAP {mode_label}"):
        audio_path, metadata = test_dataset[i]
        audio_path = str(audio_path)

        consensus_emotion = metadata["consensus_emotion"]
        session = metadata["session_number"]

        if consensus_emotion not in IEMOCAP_TO_MODEL_EMOTION or session != 5:
            skipped += 1
            continue

        emotion = IEMOCAP_TO_MODEL_EMOTION[consensus_emotion]
        text = metadata["transcription"]

        predicted_label, _probs, _confidence = model.predict(audio_path, text)

        if use_four_emotions and use_all_emotions:
            # Remap both ground-truth and predicted surprise to neutral so that
            # results are comparable with 4-class literature benchmarks.
            if consensus_emotion == "surprise":
                emotion = IEMOCAP_SURPRISE_REMAP
            if predicted_label == "surprise":
                predicted_label = IEMOCAP_SURPRISE_REMAP

        if not use_all_emotions:
            label = config.SENTIMENT_TO_IDX[config.EMOTION_TO_SENTIMENT[emotion]]
            predicted_label = config.SENTIMENT_TO_IDX[
                config.EMOTION_TO_SENTIMENT[predicted_label]
            ]
        else:
            label = iemocap_emotion_to_idx[emotion]
            predicted_label = iemocap_emotion_to_idx[predicted_label]

        preds.append(predicted_label)
        actual.append(label)

    if skipped:
        logger.info(
            f"\nSkipped {skipped} samples with unmappable 'other' consensus label."
        )

    tensor_preds = torch.Tensor(preds)
    tensor_actual = torch.Tensor(actual)

    confusion_matrix.update(tensor_preds, tensor_actual)
    accuracy.update(tensor_preds, tensor_actual)
    weighted_accuracy.update(tensor_preds, tensor_actual)
    precision_macro.update(tensor_preds, tensor_actual)
    precision_per_class.update(tensor_preds, tensor_actual)
    precision_weighted.update(tensor_preds, tensor_actual)
    recall_macro.update(tensor_preds, tensor_actual)
    recall_per_class.update(tensor_preds, tensor_actual)
    recall_weighted.update(tensor_preds, tensor_actual)
    f1_macro.update(tensor_preds, tensor_actual)
    f1_per_class.update(tensor_preds, tensor_actual)
    f1_weighted.update(tensor_preds, tensor_actual)

    # ── Overall metrics ──────────────────────────────────────────────────────
    _log_section("OVERALL METRICS")
    logger.info(f"Accuracy: {accuracy.compute():.4f}")
    logger.info(f"Weighted Accuracy: {weighted_accuracy.compute():.4f}")
    logger.info("")
    logger.info("Macro Averages (unweighted):")
    logger.info(f"  Precision: {precision_macro.compute():.4f}")
    logger.info(f"  Recall:    {recall_macro.compute():.4f}")
    logger.info(f"  F1 Score:  {f1_macro.compute():.4f}")
    logger.info("")
    logger.info("Weighted Averages (by class frequency):")
    logger.info(f"  Precision: {precision_weighted.compute():.4f}")
    logger.info(f"  Recall:    {recall_weighted.compute():.4f}")
    logger.info(f"  F1 Score:  {f1_weighted.compute():.4f}")

    # ── Per-class metrics ────────────────────────────────────────────────────
    _log_section("PER-CLASS METRICS")
    precision_vals = precision_per_class.compute()
    recall_vals = recall_per_class.compute()
    f1_vals = f1_per_class.compute()

    logger.info(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    logger.info("-" * 60)
    for i, emotion in enumerate(emotions):
        logger.info(
            f"{emotion:<12} {precision_vals[i]:.4f}       {recall_vals[i]:.4f}       {f1_vals[i]:.4f}"
        )

    # ── Class distribution ───────────────────────────────────────────────────
    _log_section("CLASS DISTRIBUTION")
    unique, counts = torch.unique(tensor_actual, return_counts=True)
    for idx, count in zip(unique, counts):
        label_name = emotions[int(idx)]
        percentage = (count / len(tensor_actual)) * 100
        logger.info(f"{label_name:<12}: {count:>4} samples ({percentage:>5.1f}%)")

    # ── Confusion matrix plot ────────────────────────────────────────────────
    fig, _ax = confusion_matrix.plot(labels=emotions, cmap="Blues")
    filename = (
        f"{model_path.stem}_iemocap_emotion_confusion_matrix"
        if use_all_emotions
        else f"{model_path.stem}_iemocap_sentiment_confusion_matrix"
    )
    save_path = config.STATS_DIR / f"{filename}.pdf"
    fig.savefig(save_path, format="pdf")
    logger.info(f"\nConfusion matrix saved to {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the emotion recognition model on MELD and/or IEMOCAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["meld", "iemocap", "both"],
        default="both",
        help="Which dataset(s) to evaluate on.",
    )
    parser.add_argument(
        "--mode",
        choices=["emotion", "sentiment", "both"],
        default="both",
        help=(
            "Whether to evaluate using full emotion classes, "
            "collapsed sentiment classes, or both."
        ),
    )
    parser.add_argument(
        "--output",
        choices=["terminal", "file", "both"],
        default="terminal",
        help="Where to write the results.",
    )
    parser.add_argument(
        "--model",
        default=config.MODEL_FILENAME,
        help="Model filename (relative to config.MODELS_DIR).",
    )
    parser.add_argument(
        "--four-emotions",
        action="store_true",
        default=False,
        help=(
            "IEMOCAP only: remap 'surprise' labels to 'neutral' so results use "
            "4 classes (neutral, joy, sadness, anger), matching past literature "
            "that excludes surprise. Has no effect on MELD or sentiment mode."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = config.MODELS_DIR / args.model
    log_path = Path(f"./{model_path.stem}_test_results.txt")

    setup_logging(args.output, log_path)

    if args.output in ("file", "both"):
        logger.info(f"Model: {model_path}")

    run_emotion = args.mode in ("emotion", "both")
    run_sentiment = args.mode in ("sentiment", "both")
    run_meld = args.dataset in ("meld", "both")
    run_iemocap = args.dataset in ("iemocap", "both")

    if run_meld:
        if run_emotion:
            test_meld(use_all_emotions=True, model_path=model_path)
        if run_sentiment:
            test_meld(use_all_emotions=False, model_path=model_path)

    if run_iemocap:
        if run_emotion:
            test_iemocap(
                use_all_emotions=True,
                model_path=model_path,
                use_four_emotions=args.four_emotions,
            )
        if run_sentiment:
            test_iemocap(
                use_all_emotions=False,
                model_path=model_path,
                use_four_emotions=args.four_emotions,
            )

    if args.output in ("file", "both"):
        logger.info(f"\nResults saved to {log_path.resolve()}")


if __name__ == "__main__":
    main()
