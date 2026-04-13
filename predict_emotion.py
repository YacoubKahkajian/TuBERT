"""
This module implements the audio pipeline that powers TuBERT's live emotion
detection. The public entry point is ``run_emotion_detection``, which is
designed to be executed on a dedicated background thread so the tkinter GUI
remains responsive.

Audio pipeline overview
-----------------------
1. **Initialization**
A PyAudio input stream is opened at 16 kHz, mono, 16-bit
PCM.  Alongside it, a Vosk speech-recognition model and a
``StreamingEmotionPredictor`` (wrapping the trained
multimodal GRU) are loaded.

2. **Voice-activity detection (VAD) loop**
On every iteration the loop reads a 1024-sample chunk
(~64 ms) from the microphone and computes its mean absolute
amplitude.  The amplitude is compared against a threshold
derived from the user-configured dBFS value:

amplitude_threshold = 32767 × 10^(dBFS / 20)

If volume > threshold, the chunk is appended to the accumulation
buffer ``full_audio_chunk`` and ``silent_chunks`` is reset to
zero.

If volume ≤ threshold and the user was previously speaking,
``silent_chunks`` is incremented and the chunk is still appended
so that the tail of the utterance is preserved. Once  ``silent_chunks``
exceeds the silence threshold count, the accumulated audio is processed.

3. **Pre-roll buffer**
A ``collections.deque`` of the last three raw chunks (``prev_frames``) is
prepended to the accumulation buffer the moment speech is first detected.
This gives the transcriber a few frames of audio context from just before
the microphone level rose, reducing clipped words and increasing
transcription accuracy.

4. **Transcription**
The concatenated utterance is passed to a freshly instantiated
``KaldiRecognizer`` (Vosk).  ``AcceptWaveform`` processes the raw PCM
bytes and ``Result()`` returns a JSON object whose ``"text"`` field
contains the recognised transcript.

5. **Emotion inference**
The raw PCM array is loaded into ``StreamingEmotionPredictor``'s ring
buffer via ``add_audio``, then ``predict_from_buffer`` writes the buffer
to a temporary WAV file, extracts MFCC + delta + delta-delta audio
features with CMVN normalisation, produces a DistilBERT text embedding
from the transcript, and runs both through the multimodal GRU to yield an
emotion label, a per-class probability dictionary, and a confidence score.

6. **Neutral confidence fallback**
If the top prediction is ``"neutral"`` *and* its confidence is below the
user-configured neutral threshold, the second-highest ranked emotion is
used instead.  This prevents the model from defaulting to neutral for
weakly-expressed emotions and allows users to customize how expressive
they want their sprite to be.

7. **Result dispatch**
The final (emotion, probs, confidence, transcript) tuple is passed to the
``on_result`` callback, which the GUI uses to update its labels and sprite.

Threading model
---------------
``run_emotion_detection`` blocks its calling thread until ``stop_event`` is
set.  It must therefore be run on a ``threading.Thread`` (daemon=True is
recommended).  All communication with the GUI thread goes through the two
callbacks (``on_result``, ``on_mic_update``) Both callbacks schedule their
tkinter mutations with ``root.after(0, ...)`` on the GUI side, so no tkinter
objects are touched directly from this thread.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import pyaudio
from dotenv import load_dotenv
from vosk import KaldiRecognizer, Model

from predictor_classes import StreamingEmotionPredictor


def run_emotion_detection(
    on_result,
    on_mic_update,
    stop_event,
    loudness_threshold,
    silence_threshold,
    neutral_confidence_threshold,
):
    """
    Run the real-time audio emotion-detection pipeline on the calling thread.
    This function blocks indefinitely, reading from the default microphone,
    until ``stop_event`` is set.  It must be called from a background thread.

    Args:
        on_result : (Callable[[str, dict[str, float], float, str], None])
            Callback invoked once per completed utterance with:
              - ``predicted_emotion`` (str)  — top emotion label.
              - ``probs`` (dict)             — per-class softmax probabilities.
              - ``confidence`` (float)       — probability of the top emotion.
              - ``transcript_text`` (str)    — Vosk transcript of the utterance.

        on_mic_update : (Callable[[bool], None])
            Callback invoked whenever the voice-activity state changes.
            Receives ``True`` when speech is first detected and ``False``
            when the utterance ends (silence threshold exceeded).

        stop_event : (threading.Event)
            When set, the VAD loop exits after its current iteration and the
            function returns, allowing the thread to terminate cleanly.

        loudness_threshold : (ttk.Scale)
            Tkinter slider whose ``.get()`` returns a *positive* number
            representing the loudness threshold magnitude.  The backend
            negates this value to convert it to a dBFS level and then derives
            a linear int16 amplitude threshold from it.

        silence_threshold : (ttk.Scale)
            Tkinter slider whose ``.get()`` returns the required silence
            duration in seconds.  Converted to a chunk count using:
            ``silence_chunks_threshold = seconds / (samples_per_chunk /
            sample_rate)``.

        neutral_confidence_threshold : (ttk.Scale)
            Tkinter slider whose ``.get()`` returns a value in the range
            0–100 (percentage).  Divided by 100 before comparison with the
            model's raw confidence score.  If the model predicts "neutral"
            with confidence below this value, the second-best emotion is
            used instead.
    """
    # Set up models
    load_dotenv()
    predictor = StreamingEmotionPredictor(device="cpu", model_path="model/e_best.pt")
    vosk_model = Model(lang="en-us")

    sample_rate = 16000
    samples_per_chunk = 1024

    # Begin recording audio
    audio_stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=samples_per_chunk,
    )
    audio_stream.start_stream()

    full_audio_chunk = []  # Accumulates PCM chunks for the current utterance.
    prev_frames = deque(maxlen=3)  # Rolling pre-roll buffer: last 3 chunks (~192 ms).
    silent_chunks = 0  # Number of consecutive below-threshold chunks seen.
    is_speaking = False  # VAD state: True while the user is actively talking.
    print("Audio stream initialized")

    while not stop_event.is_set():
        # Read a single chunk from the microphone (non-blocking on overflow).
        audio_chunk = audio_stream.read(1024, exception_on_overflow=False)
        np_audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)

        # Mean absolute amplitude as a proxy for perceived loudness.
        volume = np.mean(np.abs(np_audio_chunk))

        # Convert the dBFS threshold to a linear int16 amplitude for comparison.
        # The slider stores the magnitude as a positive number, so we negate it
        # to recover the negative dBFS value before applying the formula.
        amplitude_threshold = 32767 * (10 ** ((loudness_threshold.get() * -1) / 20))

        if volume > amplitude_threshold:
            # ── Speech detected ───────────────────────────────────────────────
            if not is_speaking:
                # Prepend the pre-roll buffer on the first loud chunk so that
                # the transcriber receives audio context from just before the
                # onset of speech, reducing clipped word errors.
                full_audio_chunk.append(np.array(list(prev_frames)).flatten())

            is_speaking = True
            on_mic_update(is_speaking)
            full_audio_chunk.append(np_audio_chunk)
            silent_chunks = 0

        else:
            # ── Silence detected ──────────────────────────────────────────────
            if is_speaking:
                silent_chunks += 1
                # Keep appending during the tail silence so the utterance
                # ending is captured and the transcriber isn't cut short.
                full_audio_chunk.append(np_audio_chunk)

                # Each chunk covers approximately 1024 / 16000 = 0.064 s, so
                # dividing the desired silence duration by that gives the
                # number of consecutive silent chunks to wait for.
                silence_chunks_threshold = silence_threshold.get() / (
                    samples_per_chunk / sample_rate
                )
                if (
                    silent_chunks > silence_chunks_threshold
                    and len(full_audio_chunk) > 0
                ):
                    # ── Process the completed utterance ───────────────────────
                    # Concatenate all accumulated chunks into one contiguous array.
                    np_full_audio_chunk = np.concatenate(full_audio_chunk, axis=0)
                    full_audio_chunk = []
                    is_speaking = False
                    on_mic_update(is_speaking)

                    # Load the utterance into the predictor's ring buffer so
                    # that predict_from_buffer can write it to a temp WAV file.
                    predictor.add_audio(np_full_audio_chunk)

                    # Transcribe the utterance using Vosk.
                    rec = KaldiRecognizer(vosk_model, 16000)
                    rec.AcceptWaveform(np_full_audio_chunk.tobytes())
                    transcript_text = json.loads(rec.Result())
                    transcript_text = transcript_text["text"]

                    # Run multimodal emotion inference with audio features + text embedding.
                    emotion, probs, confidence = predictor.predict_from_buffer(
                        transcript_text
                    )

                    # Neutral confidence fallback: if the model picks "neutral"
                    # but with low confidence, substitute the next best emotion
                    # so borderline utterances get a more expressive result.
                    sorted_probs = sorted(
                        probs.items(), key=lambda x: x[1], reverse=True
                    )
                    if (
                        confidence < (neutral_confidence_threshold.get() / 100)
                        and emotion == "neutral"
                    ):
                        emotion, confidence = sorted_probs[1]

                    on_result(emotion, probs, confidence, transcript_text)

        # Always update the pre-roll buffer with the current chunk, whether
        # it was loud or silent, so it reflects the most recent audio context.
        prev_frames.append(np_audio_chunk)
