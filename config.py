"""
This module is the single source of truth for every constant used across
the project. Data paths, audio processing parameters, model architecture
hyperparameters, training settings, emotion/sentiment label maps, and
real-time streaming thresholds live here.

Importing this module from any script is safe regardless of the working
directory because all filesystem paths are derived from the absolute
location of this file via ``PROJECT_ROOT``.
"""

from pathlib import Path

# Absolute path to the project root (the directory this file lives in).
# All other paths are derived from it so that `import config` works correctly
# regardless of the working directory the script is launched from.
PROJECT_ROOT = Path(__file__).parent.resolve()

# ── Data paths ────────────────────────────────────────────────────────────────
# Raw MELD dataset directory (contains train/dev/test CSV files and audio clips).
MELD_ROOT = PROJECT_ROOT / "model" / "setup" / "data" / "MELD"
# Output directory for preprocessed feature files (NumPy arrays, metadata CSVs).
PREPROCESSED_ROOT = PROJECT_ROOT / "model" / "setup" / "data" / "preprocessed"
# Raw IEMOCAP dataset directory (session-based folder structure).
IEMOCAP_ROOT = PROJECT_ROOT / "model" / "setup" / "data" / "IEMOCAP"
# Top-level model directory; trained checkpoints are stored here.
MODELS_DIR = PROJECT_ROOT / "model"
# Directory for evaluation outputs such as confusion matrices and ablation results.
STATS_DIR = PROJECT_ROOT / "model" / "stats"
# File name of the model to use for evaluations
MODEL_FILENAME = "tubert.pt"

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000  # Sampling rate in Hz. Matches both MELD and microphone input.
N_MFCC = 40  # Number of Mel-frequency cepstral coefficients per frame.
N_MELS = 128  # Number of mel filter-bank bands used for the mel spectrogram.
HOP_LENGTH = 160  # STFT hop in samples (10 ms frame shift at 16 kHz)
WIN_LENGTH = 400  # STFT window in samples (25 ms frame length at 16 kHz)
MAX_AUDIO_LENGTH = (
    10.0  # Clips longer than this in seconds are truncated at feature extraction time.
)

# ── Text settings ─────────────────────────────────────────────────────────────
# Pre-trained DistilBERT checkpoint fine-tuned on emotion data, used as the
# text encoder.  Produces 768-dimensional [CLS] embeddings.
DISTILBERT_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
MAX_TEXT_LENGTH = 128  # Maximum number of WordPiece tokens fed to the text encoder.

# ── Emotion labels ────────────────────────────────────────────────────────────
# Ordered list of emotion classes used for MELD training and inference.
# The index of each emotion in this list corresponds to the model's output logit index.
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness"]

# Subset of emotions used when training and testing on IEMOCAP
EMOTIONS_IEMOCAP = ["neutral", "joy", "anger", "sadness"]

# Bidirectional lookup tables between emotion strings and integer class indices.
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

# Ordered list of sentiment classes, coarser than emotions).
SENTIMENTS = ["neutral", "positive", "negative"]

# Bidirectional lookup tables between sentiment strings and integer class indices.
SENTIMENT_TO_IDX = {sentiment: idx for idx, sentiment in enumerate(SENTIMENTS)}
IDX_TO_SENTIMENT = {idx: sentiment for sentiment, idx in SENTIMENT_TO_IDX.items()}

# Maps each fine-grained emotion to its coarse sentiment polarity.
# Note: "surprise" is mapped to "negative" to align with MELD's annotation
# convention, where surprise tends to co-occur with negative contexts.
EMOTION_TO_SENTIMENT = {
    "neutral": "neutral",
    "joy": "positive",
    "surprise": "negative",
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
}

# ── Model architecture ────────────────────────────────────────────────────────
# Input dimensionality of the audio branch: MFCC coefficients concatenated
# with their first- and second-order temporal derivatives (delta and delta-delta).
AUDIO_FEATURE_DIM = N_MFCC * 3  # = 120
# Dimensionality of the [CLS] embedding produced by DistilBERT.
TEXT_EMBEDDING_DIM = 768
# Number of hidden units in each direction of the bidirectional GRU.
GRU_HIDDEN_DIM = 256
# Number of stacked GRU layers in the audio encoder.
GRU_NUM_LAYERS = 2
# Dropout probability applied between GRU layers.
GRU_DROPOUT = 0.3
# When True, the GRU runs in both forward and backward directions; the
# outputs are concatenated, doubling the effective hidden dimension to 512.
BIDIRECTIONAL = True
# When True, a learned attention layer pools the GRU time-step outputs into
# a single context vector instead of using the final hidden state alone.
USE_ATTENTION = True

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
# Number of validation epochs with no improvement before training is stopped early.
EARLY_STOPPING_PATIENCE = 10
# Compute device for PyTorch tensors. Override with "mps" on Apple Silicon
# or "cuda" on machines with an NVIDIA GPU.
DEVICE = "cpu"

# ── Data splits ───────────────────────────────────────────────────────────────
# Names of the three standard data partitions used by MELD (and mirrored for IEMOCAP).
SPLITS = ["train", "dev", "test"]

# ── Streaming / real-time settings ───────────────────────────────────────────
# Number of PyAudio chunks accumulated before the buffer is considered full.
CHUNKS_PER_BUFFER = 20
# Minimum loudness (dBFS) required to consider audio as speech.
# Valid range is -60 (very quiet) to 0 (maximum digital level).
LOUDNESS_THRESHOLD = -30
# Duration of continuous silence (seconds) required before a completed
# utterance is flushed to the emotion predictor.
SILENCE_THRESHOLD = 0.5
# Minimum model confidence required to accept a "neutral" prediction as-is.
# If the top prediction is "neutral" but its probability is below this
# threshold, the second-highest-probability emotion is displayed instead.
NEUTRAL_CONFIDENCE_THRESHOLD = 0.65
