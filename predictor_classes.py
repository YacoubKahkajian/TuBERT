"""
Predictor classes for single-file and streaming real-time emotion inference.
This module provides two public classes:

  * ``EmotionPredictor``: for offline or batch inference over individual audio
  files paired with their transcriptions.

  * ``StreamingEmotionPredictor``: for real-time microphone input, using a
  fixed-length ring buffer and periodic prediction calls.

Both classes use the multimodal GRU + DistilBERT architecture
(``MultiModalEmotionRNN``) and expect audio features that have been
Cepstral Mean and Variance Normalised.
"""

from pathlib import Path

import numpy as np
import torch

import config
from model.setup.feature_extractor import (
    AudioFeatureExtractor,
    TextFeatureExtractor,
    apply_cmvn,
)
from model.setup.rnn import MultiModalEmotionRNN


class EmotionPredictor:
    """
    Wraps ``MultiModalEmotionRNN`` to provide a convenient high-level
    inference API for audio files paired with their text transcriptions.
    The model fuses a GRU-encoded audio stream with a DistilBERT text
    embedding to classify the utterance into one of the emotions defined
    in ``config.EMOTIONS``.

    Two inference paths are available:

    * ``predict(audio_path, transcription)``: accepts a WAV file path and
      handles all feature extraction and normalisation internally.
    * ``predict_from_features(audio_features, text_embedding)``: accepts
      pre-extracted numpy arrays, which is more efficient for batch
      processing where features have already been computed.

    CMVN normalisation is applied inside ``AudioFeatureExtractor.extract()``,
    so both training-time preprocessing and inference-time feature extraction
    go through the same normalisation path automatically.
    """

    def __init__(self, model_path="models/best.pt", device=config.DEVICE):
        """
        Initialise feature extractors, load the trained model, and set it
        to evaluation mode. The checkpoint is expected to be a dict
        containing at least the key ``"model_state_dict"``, which is
        the format saved by the training loop.

        Args:
            model_path : (str | Path)
                Path to the trained model checkpoint (``.pt`` file).
                Defaults to ``"models/best.pt"``.
            device : (str)
                PyTorch device string - ``'cpu'``, ``'cuda'``, or ``'mps'``.
                Defaults to ``config.DEVICE``.
        """
        self.device = device
        self.audio_extractor = AudioFeatureExtractor()
        self.text_extractor = TextFeatureExtractor(device=device)

        self.model = MultiModalEmotionRNN().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_path, transcription):
        """
        Predict emotion from an audio file and its transcription.
        Extracts MFCC-based audio features via ``AudioFeatureExtractor``
        and a DistilBERT embedding via ``TextFeatureExtractor``, then runs
        a single forward pass through ``MultiModalEmotionRNN``.

        Args:
            audio_path : (str | Path)
                Path to a WAV audio file.
            transcription : (str)
                Text transcription of the spoken utterance.

        Returns:
            emotion : (str)
                The predicted emotion label (e.g. ``"joy"``, ``"anger"``).
            probabilities : (dict[str, float])
                Softmax probability for every emotion in ``config.EMOTIONS``,
                keyed by emotion label.
            confidence : (float)
                Softmax probability of the predicted (argmax) emotion.
        """
        audio_features = self.audio_extractor.extract(audio_path)
        text_embedding = self.text_extractor.extract(transcription)

        audio_tensor = (
            torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
        )  # (1, time, features)
        text_tensor = (
            torch.FloatTensor(text_embedding).unsqueeze(0).to(self.device)
        )  # (1, 768)
        length_tensor = torch.LongTensor([audio_features.shape[0]]).to(
            self.device
        )  # (1,)

        logits = self.model(audio_tensor, text_tensor, length_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = np.argmax(probs)
        emotion = config.IDX_TO_EMOTION[int(pred_idx)]
        confidence = probs[pred_idx]

        prob_dict = {
            config.IDX_TO_EMOTION[i]: float(probs[i])
            for i in range(len(config.EMOTIONS))
        }

        return emotion, prob_dict, confidence

    @torch.no_grad()
    def predict_from_features(self, audio_features, text_embedding):
        """
        Predict emotion from pre-extracted features.

        This method skips feature extraction entirely and is therefore faster
        than ``predict()`` when features have already been computed (e.g.
        during batch evaluation over a pre-processed dataset).

        If you are passing raw (un-normalised) audio features, apply
        ``apply_cmvn()`` before calling this method so that the input
        distribution matches what the model was trained on::

            from model.setup.feature_extractor import apply_cmvn
            audio_features = apply_cmvn(audio_features)

        Args:
            audio_features : (np.ndarray)
                Shape ``(time, feature_dim)`` - MFCC features that have already been CMVN-normalised.
            text_embedding : (np.ndarray)
                Shape ``(768,)`` - DistilBERT [CLS] embedding for the utterance.

        Returns:
            emotion : (str)
                The predicted emotion label.
            probabilities : (dict[str, float])
                Softmax probability for every emotion in ``config.EMOTIONS``.
            confidence : (float)
                Softmax probability of the predicted (argmax) emotion.
        """
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(self.device)
        text_tensor = torch.FloatTensor(text_embedding).unsqueeze(0).to(self.device)
        length_tensor = torch.LongTensor([audio_features.shape[0]]).to(self.device)

        logits = self.model(audio_tensor, text_tensor, length_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = np.argmax(probs)
        emotion = config.IDX_TO_EMOTION[int(pred_idx)]
        confidence = probs[pred_idx]
        prob_dict = {
            config.IDX_TO_EMOTION[i]: float(probs[i])
            for i in range(len(config.EMOTIONS))
        }

        return emotion, prob_dict, confidence


class StreamingEmotionPredictor:
    """
    Streaming emotion predictor designed for real-time microphone input.

    Internally maintains a fixed-length **ring buffer** (``audio_buffer``) of
    float32 audio samples.  Incoming audio chunks are appended to the tail of
    the buffer via ``np.roll``, displacing the oldest samples.

    When a prediction is requested via ``predict_from_buffer()``, the full
    contents of the ring buffer are written to a temporary WAV file on disk.
    ``EmotionPredictor.predict()`` then processes that file and the
    temporary file is deleted immediately after inference completes.

    The predicted emotion label from every call to ``predict_from_buffer()``
    is appended to ``emotion_history``, providing a running log of all
    predictions made during the session.
    """

    def __init__(
        self,
        model_path="model/best.pt",
        device=config.DEVICE,
        buffer_duration=3.0,
        update_interval=1.0,
    ):
        """
        Initialise the ring buffer, underlying predictor, and session state.

        Args:
            model_path : (str | Path)
                Path to the trained model checkpoint. Defaults to ``"model/best.pt"``.
            device : (str)
                PyTorch device string — ``'cpu'``, ``'cuda'``, or``'mps'``.  Defaults to ``config.DEVICE``.
            buffer_duration : (float)
                Length of the ring buffer in seconds. Determines how many samples of audio are retained for the
                next prediction.  Defaults to ``3.0``.
            update_interval : (float)
                Intended minimum time between successive prediction calls, in seconds.  This value is stored as an
                attribute for use by callers; it is *not* enforced internally by this class.  Defaults to ``1.0``.

        Attributes:
            predictor : (EmotionPredictor)
                The underlying single-file predictor used to run each inference call.
            buffer_duration : (float)
                Ring-buffer duration in seconds, as passed to the constructor.
            update_interval : (float)
                Minimum seconds between predictions, as passed to the constructor.
            sample_rate : (int)
                Audio sample rate taken from ``config.SAMPLE_RATE`` (default 16000 Hz).
            buffer_size : (int)
                Number of samples in the ring buffer (``buffer_duration * sample_rate``).
            audio_buffer : (np.ndarray)
                Float32 array of shape ``(buffer_size,)`` holding the most recent audio samples.
                Initialised to silence (zeros).
            current_emotion : (str)
                Emotion label from the most recent prediction.  Initialised to ``"neutral"``.
            current_confidence : (float)
                Softmax confidence of the most recent prediction.  Initialised to ``0.0``.
            emotion_history : (list[str])
                Ordered list of all emotion labels predicted since this instance was created.
                Each call to ``predict_from_buffer()`` appends one entry.
        """
        self.predictor = EmotionPredictor(model_path, device)
        self.buffer_duration = buffer_duration
        self.update_interval = update_interval
        self.sample_rate = config.SAMPLE_RATE

        self.buffer_size = int(buffer_duration * self.sample_rate)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.emotion_history = []

    def add_audio(self, audio_chunk):
        """
        Append an incoming audio chunk to the ring buffer.

        The buffer behaves as a sliding window over the most recent
        ``buffer_duration`` seconds of audio.

        Args:
            audio_chunk : (np.ndarray)
                1-D float32 array of audio samples at ``self.sample_rate``Hz.
        """
        chunk_size = len(audio_chunk)

        if chunk_size >= self.buffer_size:
            self.audio_buffer = audio_chunk[-self.buffer_size :]
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -chunk_size)
            self.audio_buffer[-chunk_size:] = audio_chunk

    def predict_from_buffer(self, transcription):
        """
        Run emotion inference over the current ring-buffer contents.

        The buffer is serialised to a temporary WAV file.  The
        file is deleted with ``Path.unlink()`` immediately after
        inference returns, regardless of whether the prediction succeeds
        to keep TuBERT moving.

        After a successful prediction ``self.current_emotion`` is updated to
        the new label and ``self.current_confidence`` is updated to the new
        confidence score. The new label is appended to ``self.emotion_history``.

        Args:
            transcription : (str)
                Text transcription of the current utterance

        Returns:
            emotion : (str)
                Predicted emotion label.
            probabilities : (dict[str, float])
                Softmax probability for every emotion in ``config.EMOTIONS``.
            confidence : (float)
                Softmax probability of the predicted (argmax) emotion.
        """
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, self.audio_buffer, self.sample_rate)
            tmp_path = tmp.name

        emotion, probs, confidence = self.predictor.predict(tmp_path, transcription)

        Path(tmp_path).unlink()

        self.current_emotion = emotion
        self.current_confidence = confidence
        self.emotion_history.append(emotion)

        return emotion, probs, confidence
