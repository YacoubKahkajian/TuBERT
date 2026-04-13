"""
Feature extraction for audio and text
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import config

# Minimum number of frames to compute reliable per-utterance CMVN stats.
# Utterances shorter than this fall back to zero-mean / unit-variance
# (i.e. a no-op that at least keeps the scale consistent).
_CMVN_MIN_FRAMES = 50  # ~0.5 s at 16 kHz with hop_length=160


def apply_cmvn(features: np.ndarray) -> np.ndarray:
    """
    Apply per-utterance Cepstral Mean and Variance Normalisation (CMVN).
    Subtracts the utterance mean and divides by the utterance std along the
    time axis so that each feature dimension has zero mean and unit variance
    within the utterance.  This reduces the effect of channel/speaker/domain
    differences, helpful when generalising across datasets.

    Args:
        features : (np.ndarray)
            Array of shape (time_steps, feature_dim) containing raw features.

    Returns:
        features : (np.ndarray)
            Normalised array of the same shape as the input.
    """
    if features.shape[0] < _CMVN_MIN_FRAMES:
        return features

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero for silent/constant dims
    return (features - mean) / std


class AudioFeatureExtractor:
    """Extract MFCC + delta + delta-delta features from audio.

    Each call to :meth:`extract` loads a WAV file or a pre-loaded
    waveform, resamples to ``config.SAMPLE_RATE`` if necessary, converts to
    mono, computes ``config.N_MFCC`` Mel-frequency cepstral coefficients,
    appends their first- and second-order temporal derivatives, and finally
    applies per-utterance CMVN via :func:`apply_cmvn`.

    The resulting feature matrix has shape ``(time_steps, config.N_MFCC * 3)``
    and can be fed directly into the multimodal RNN as the ``audio_features``
    argument.
    """

    def extract(self, audio_path):
        """
        Extract MFCC + delta + delta-delta features from a single audio source.

        Accepts a file path, a raw NumPy waveform array, or a pre-loaded
        ``torch.Tensor``.  Resamples to ``config.SAMPLE_RATE`` if needed,
        downmixes stereo to mono, computes MFCCs, stacks their first- and
        second-order deltas, and applies per-utterance CMVN.

        Args:
            audio_path : (str | np.ndarray | torch.Tensor)
                - ``str``: path to a WAV file loaded with ``torchaudio``.
                - ``np.ndarray``: raw waveform, ``config.SAMPLE_RATE`` is assumed.
                - ``torch.Tensor``: pre-loaded waveform tensor; ``config.SAMPLE_RATE``
                  is assumed.

        Returns:
            features : (np.ndarray)
                Array of shape ``(time_steps, n_mfcc * 3)`` containing the
                CMVN-normalised concatenation of [MFCC, delta, delta-delta].

        Raises:
            ValueError
                If ``audio_path`` is not one of the three supported types.
        """

        # Load audio
        if type(audio_path) is str:
            waveform, sr = torchaudio.load(audio_path)
        elif type(audio_path) is np.ndarray:
            waveform = torch.from_numpy(audio_path)
            sr = config.SAMPLE_RATE
        elif type(audio_path) is torch.Tensor:
            waveform = audio_path
            sr = config.SAMPLE_RATE
        else:
            raise ValueError("Unsupported audio type")

        # Resample if necessary
        if sr != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Compute MFCCs + deltas
        mfcc_transform_object = torchaudio.transforms.MFCC(
            sample_rate=config.SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            melkwargs={
                "n_fft": 512,
                "n_mels": config.N_MELS,
                "hop_length": config.HOP_LENGTH,
                "win_length": config.WIN_LENGTH,
            },
        )
        mfcc = mfcc_transform_object(waveform)
        mfcc = mfcc.squeeze(0)

        delta = torchaudio.functional.compute_deltas(mfcc)
        delta_delta = torchaudio.functional.compute_deltas(delta)

        # Concatenate [MFCC, delta, delta-delta] → (feature_dim, time)
        features = torch.cat([mfcc, delta, delta_delta], dim=0)

        # Transpose to (time, features)
        features = features.transpose(0, 1).numpy()

        features = apply_cmvn(features)

        return features

    def extract_batch(self, audio_paths):
        """
        Extract MFCC + delta + delta-delta features for a list of audio files.

        Args:
            audio_paths : (list[str])
                Paths to WAV files on disk.

        Returns:
            features : (list[np.ndarray])
                One array per file, each of shape ``(time_steps, n_mfcc * 3)``.
                Time-steps may differ across files; use ``collate_fn`` to pad
                a batch.
        """
        return [
            self.extract(path)
            for path in tqdm(audio_paths, desc="Extracting audio features")
        ]


class TextFeatureExtractor:
    """
    Extract fixed-length sentence embeddings using DistilBERT.
    Loads a pretrained DistilBERT tokenizer and model (default:
    ``config.DISTILBERT_MODEL``) and extracts the hidden state of the
    [CLS] token at the final layer as a 768-dimensional embedding for each
    input utterance.
    """

    def __init__(self, model_name=config.DISTILBERT_MODEL, device=config.DEVICE):
        """
        Load the DistilBERT tokenizer and model.

        Args:
            model_name : (str)
                HuggingFace model identifier for the tokenizer and model
                weights.  Defaults to ``config.DISTILBERT_MODEL``.
            device : (str)
                PyTorch device string — ``"cpu"``, ``"cuda"``, or ``"mps"``.
                Defaults to ``config.DEVICE``.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, text):
        """
        Extract a DistilBERT [CLS]-token embedding for a single utterance.

        Tokenises the input string (padding + truncation to
        ``config.MAX_TEXT_LENGTH`` tokens), runs it through DistilBERT, and
        returns the last-layer hidden state at position 0 as a
        1-D NumPy array.

        Args:
            text : (str)
                The utterance transcription to embed.

        Returns:
            embedding : (np.ndarray)
                Array of shape ``(768,)`` containing the [CLS]-token hidden
                state from the final DistilBERT layer.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH,
        ).to(self.device)

        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, texts):
        """
        Tokenises all strings together in a single batched call (with padding
        and truncation), runs them through DistilBERT in one forward pass, and
        returns the last-layer [CLS] hidden states as a 2-D NumPy array.

        Args:
            texts : (list[str])
                Utterance transcriptions to embed.

        Returns:
            embeddings : (np.ndarray)
                Array of shape ``(batch_size, 768)`` containing one
                768-dimensional [CLS]-token embedding per input string.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH,
        ).to(self.device)

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings.cpu().numpy()


def normalize_features(features, mean=None, std=None):
    """
    Apply global mean/variance normalisation across a feature set.

    This is a dataset-level normalisation utility used in earlier
    version of the TuBERT model, kept for reference.
    Use :func:`apply_cmvn` instead when training your own TuBERT.

    Args:
        features : (list[np.ndarray] | np.ndarray)
            Either a list of ``(time_steps, feature_dim)`` arrays or a single
            concatenated ``(total_time_steps, feature_dim)`` array.
        mean : (np.ndarray | None)
            Pre-computed per-feature mean of shape ``(feature_dim,)``.
            Computed from ``features`` if ``None``.
        std : (np.ndarray | None)
            Pre-computed per-feature standard deviation of shape
            ``(feature_dim,)``.  Computed from ``features`` if ``None``.
            Dimensions with zero variance are set to 1.0 to avoid division
            by zero.

    Returns:
        normalized_features : (np.ndarray)
            Zero-mean, unit-variance array of shape
            ``(total_time_steps, feature_dim)``.
        mean : (np.ndarray)
            Per-feature mean used for normalisation.
        std : (np.ndarray)
            Per-feature std used for normalisation.
    """
    if isinstance(features, list):
        features = np.concatenate(features, axis=0)
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
        std[std == 0] = 1.0

    normalized = (features - mean) / std
    return normalized, mean, std
