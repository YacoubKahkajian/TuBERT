"""
This module builds and runs the TuBERT settings window using tkinter. It is
responsible for:

  * Presenting three user-adjustable detection sliders:
      - Loudness Threshold: the microphone amplitude level above which
        TuBERT considers the user to be speaking.
      - Silence Threshold: the duration of quiet audio that must elapse
        after speech before the recorded utterance is considered ready.
        to be processed.
      - Neutral Threshold: the minimum confidence required for the model's
        "neutral" prediction to be accepted. If the model predicts neutral
        below this confidence, the second-highest emotion is displayed instead.

  * Providing a sprite picker UI that lets the user assign custom image files
    to each emotion (neutral, joy, angry, sad, surprise) for both the
    open-mouth and closed-mouth animation states. Selected files are copied
    into the local ``sprites/`` directory with a standardised naming scheme.

  * Displaying live inference results (predicted emotion and confidence) as
    they arrive from the background detection thread.

  * Launching the sprite display window (via ``sprite_window``) as a
    ``tk.Toplevel`` overlay that renders the appropriate character sprite in
    real time.

  * Starting a daemon ``threading.Thread`` that runs ``run_emotion_detection``
    in the background. That thread reads from the microphone, performs voice-
    activity detection, transcribes speech with Vosk, and infers emotions with
    ``StreamingEmotionPredictor``. It communicates results back to the main
    thread exclusively through the ``on_result`` and ``on_mic_update``
    callbacks, which schedule all tkinter state updates via ``root.after(0,
    ...)`` to remain thread-safe.

  * Signalling the background thread to stop cleanly via a
    ``threading.Event`` once the tkinter main loop exits.
"""

import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from predict_emotion import run_emotion_detection
from sprite_window import sprite_window

# Maximum pixel width at which descriptive label text will wrap inside the
# settings window. Keeps long hint strings from stretching the layout.
text_wrap = 300

# ── Root window ───────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("TuBERT Settings")

# Single resizable frame that fills the root window. All widgets are placed
# inside it using the grid geometry manager.
mainframe = ttk.Frame(root, padding=12)
mainframe.grid(column=0, row=0, sticky="nsew")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)


def update_sliders(*args):
    """Refresh the labels that sit beside each slider.

    Called automatically by each ``ttk.Scale`` widget's ``command`` callback
    whenever its value changes. Converts the raw slider positions into
    display-friendly strings.

    Note that for the loudness threshold label, the slider stores a positive
    value (0–60) despite it being negated here for the label. See below as
    to why this is the case.
    """
    curr_loudness_threshold.set(f"{loudness_threshold_slider.get() * -1:.0f} dBFS")
    curr_silence_threshold.set(f"{silence_threshold_slider.get():.2f}s")
    curr_neutral_threshold.set(f"{neutral_threshold_slider.get():.0f}%")


# ── Settings section ──────────────────────────────────────────────────────────
ttk.Label(mainframe, text="Settings", font=("SF Pro", 15, "bold")).grid(
    column=0, row=0, sticky="w"
)

ttk.Label(mainframe, text="Loudness Threshold").grid(column=0, row=1, sticky="w")
curr_loudness_threshold = tk.StringVar(value="-30 dBFS")

# ttk.Scale knobs do not move when the slider range includes negative values,
# so the range is kept positive (0–60) and the displayed value is negated
# to present the threshold as a conventional negative dBFS number to the user.
loudness_threshold_slider = ttk.Scale(
    mainframe, from_=0, to=60, value=30, command=update_sliders
)

loudness_threshold_slider.grid(column=1, row=1, sticky="ew", columnspan=2)
ttk.Label(mainframe, textvariable=curr_loudness_threshold).grid(
    column=4, row=1, sticky="w"
)
ttk.Label(
    mainframe,
    text="TuBERT will begin recording audio when loudness exceeds this threshold.",
    font=("SF Pro", 10),
    foreground="#888888",
    wraplength=text_wrap,
).grid(row=2, sticky="w", pady=(0, 5))

ttk.Label(mainframe, text="Silence Threshold").grid(column=0, row=3, sticky="w")
curr_silence_threshold = tk.StringVar(value="0.50s")
silence_threshold_slider = ttk.Scale(
    mainframe, from_=0.1, to=5.0, value=0.5, command=update_sliders
)
silence_threshold_slider.grid(column=1, row=3, sticky="ew", columnspan=2)
ttk.Label(mainframe, textvariable=curr_silence_threshold).grid(
    column=4, row=3, sticky="w"
)
ttk.Label(
    mainframe,
    text="After you stop speaking, TuBERT will process what you said if the pause is at least this long.",
    font=("SF Pro", 10),
    foreground="#888888",
    wraplength=text_wrap,
).grid(row=4, sticky="w", pady=(0, 5))

ttk.Label(mainframe, text="Neutral Threshold").grid(column=0, row=5, sticky="w")
neutral_threshold_slider = ttk.Scale(
    mainframe, from_=0, to=100, value=65, command=update_sliders
)
curr_neutral_threshold = tk.StringVar(value="65%")
neutral_threshold_slider.grid(column=1, row=5, sticky="ew", columnspan=2)
ttk.Label(mainframe, textvariable=curr_neutral_threshold).grid(
    column=4, row=5, sticky="w"
)
ttk.Label(
    mainframe,
    text="If TuBERT predicts neutral, it will display the neutral sprite if its confidence is above this threshold. Otherwise, it will display its second best guess.",
    font=("SF Pro", 10),
    foreground="#888888",
    wraplength=text_wrap,
).grid(row=6, sticky="w", pady=(0, 5))

ttk.Separator(mainframe, orient="horizontal").grid(
    row=7, column=0, columnspan=4, sticky="ew", pady=(4, 8)
)

# ── Sprites section ───────────────────────────────────────────────────────────
ttk.Label(mainframe, text="Sprites", font=("SF Pro", 15, "bold")).grid(
    column=0, row=8, sticky="w", columnspan=3
)
ttk.Label(
    mainframe,
    text="Choose image files for each emotion. Each emotion has an open-mouth and a closed-mouth variant.",
    font=("SF Pro", 10),
    foreground="#888888",
    wraplength=text_wrap,
).grid(row=9, sticky="w", pady=(0, 8), columnspan=3)

# The five emotions that TuBERT can predict and display sprites for.
EMOTIONS = ["neutral", "joy", "angry", "sad", "surprise"]

# Header row for the sprite picker table.
ttk.Label(mainframe, text="Emotion", font=("SF Pro", 11, "bold")).grid(
    column=0, row=10, sticky="w", pady=(0, 4)
)
ttk.Label(mainframe, text="Open Mouth", font=("SF Pro", 11, "bold")).grid(
    column=1, row=10, sticky="w", pady=(0, 4)
)
ttk.Label(mainframe, text="Closed Mouth", font=("SF Pro", 11, "bold")).grid(
    column=2, row=10, sticky="w", pady=(0, 4)
)


def make_browse(emotion, mouth_open):
    """
    Create and return a file-browse callback for a single sprite slot.

    Uses a closure to capture ``emotion`` and ``mouth_open`` so that each
    Browse button in the sprite table targets the correct destination file
    without requiring global state.

    When the returned callback is invoked (i.e. when the user clicks a Browse
    button), it opens a file-picker dialog filtered to common image formats.
    If the user selects a file, it is copied into the ``sprites/`` directory
    under a standardised name:  ``<emotion>_<open|closed><ext>``
    (e.g. ``joy_open.png``).  The sprite window reads files from that
    directory using the same naming convention, so the new sprite takes effect
    immediately on the next render cycle.

    Args:
        emotion : (str)
            The emotion label this sprite slot belongs to
        mouth_open : (bool)
            ``True`` for the open-mouth variant, ``False`` for
            the closed-mouth variant.

    Returns:
        Callable[[], None]
            A zero-argument callback suitable for passing as
            a ``ttk.Button`` ``command``.
    """

    def _browse():
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp")],
        )
        if path:
            path = Path(path)
            file_type = path.suffix
            open_or_closed = "open" if mouth_open else "closed"
            # Destination follows the convention expected by sprite_window:
            # sprites/<emotion>_<open|closed><ext>
            copy_path = Path("sprites") / f"{emotion}_{open_or_closed}{file_type}"
            shutil.copyfile(path, copy_path)

    return _browse


# Build one table row per emotion. Each row contains the emotion name label
# plus Browse buttons for the open-mouth and closed-mouth sprite slots.
# Rows are spaced 3 grid rows apart to leave room for future path-hint labels.
for i, emotion in enumerate(EMOTIONS):
    row = (
        11 + i * 3
    )  # 3 rows per emotion: label+pickers, path hint open, path hint closed

    # Emotion label
    ttk.Label(mainframe, text=emotion.capitalize()).grid(column=0, row=row, sticky="w")

    # ── Open mouth ────────────────────────────────────────────────────────────
    open_frame = ttk.Frame(mainframe)
    open_frame.grid(column=1, row=row, sticky="ew", padx=(0, 6))
    open_frame.columnconfigure(0, weight=1)

    ttk.Button(open_frame, text="Browse…", command=make_browse(emotion, True)).grid(
        column=1, row=0, sticky="w", padx=(4, 0)
    )

    # ── Closed mouth ──────────────────────────────────────────────────────────
    closed_frame = ttk.Frame(mainframe)
    closed_frame.grid(column=2, row=row, sticky="ew")
    closed_frame.columnconfigure(0, weight=1)

    ttk.Button(closed_frame, text="Browse…", command=make_browse(emotion, False)).grid(
        column=1, row=0, sticky="w", padx=(4, 0)
    )

# Calculate the grid row that immediately follows the last sprite row so that
# the separator and results section can be placed directly beneath it.
sprite_end_row = 11 + len(EMOTIONS) * 3
ttk.Separator(mainframe, orient="horizontal").grid(
    row=sprite_end_row, column=0, columnspan=4, sticky="ew", pady=(8, 8)
)

# ── Results section ───────────────────────────────────────────────────────────
# These labels are bound to tkinter StringVars that are updated in real time
# by the on_result callback as emotion predictions arrive from the background
# thread.
ttk.Label(mainframe, text="Results", font=("SF Pro", 15, "bold")).grid(
    column=0, row=sprite_end_row + 1, sticky="w"
)
ttk.Label(mainframe, text="Predicted Emotion").grid(
    column=0, row=sprite_end_row + 2, sticky="w"
)
emotion = tk.StringVar()
emotion_label = ttk.Label(mainframe, textvariable=emotion)
emotion_label.grid(column=1, row=sprite_end_row + 2, sticky="w")

ttk.Label(mainframe, text="Confidence").grid(
    column=0, row=sprite_end_row + 3, sticky="w"
)
confidence = tk.StringVar()
confidence_label = ttk.Label(mainframe, textvariable=confidence)
confidence_label.grid(column=1, row=sprite_end_row + 3, sticky="w")

# "Speaking..." indicator — shown while the VAD loop considers the user active;
# cleared automatically when silence is detected.
talking = tk.StringVar()
talking_label = ttk.Label(mainframe, textvariable=talking)
talking_label.grid(column=0, row=sprite_end_row + 4, sticky="w")

# ── Sprite window & background thread ────────────────────────────────────────
# Open the sprite display window and obtain its change_sprite() callable.
# sprite_window() creates a tk.Toplevel attached to root so its lifetime is
# tied to the main window.
change_sprite = sprite_window(root)

# Tracks the most recently confirmed emotion so that on_mic_update can keep
# the correct sprite on screen while toggling only the mouth-open state.
prev_emotion = ""


def on_result(predicted_emotion, probs, predicted_confidence, transcript_text):
    """
    Handle a completed emotion prediction from the background thread.

    Called by ``run_emotion_detection`` after each utterance has been
    transcribed and classified. All tkinter mutations are scheduled via
    ``root.after(0, ...)`` so they execute on the main thread, keeping the UI
    thread-safe.

    Updates the settings window's Results labels and instructs the sprite
    window to display the closed-mouth sprite for the predicted emotion
    (mouth closed because the user has just finished speaking).

    Args:
        predicted_emotion : (str)
            The top emotion label predicted by the model. May already have
            been overridden to the second-best guess if the neutral confidence
            fell below the configured threshold.
        probs : (dict[str, float])
            Mapping of every emotion label to its softmax probability for this
            utterance. Functional but currently unused.
        predicted_confidence : (float)
            Probability of the returned ``predicted_emotion``.
        transcript_text : (str)
            Vosk transcript of the utterance. Functional but currently unused.
    """
    root.after(0, lambda: emotion.set(predicted_emotion))
    root.after(0, lambda: confidence.set(f"{predicted_confidence * 100:.2f}%"))
    # Show the closed-mouth sprite — the user has finished speaking.
    root.after(0, lambda: change_sprite(predicted_emotion, False))
    global prev_emotion
    prev_emotion = predicted_emotion


def on_mic_update(is_talking):
    """Handle a voice-activity state change from the background thread.

    Called by ``run_emotion_detection`` whenever the user transitions between
    speaking and silent. Like ``on_result``, all tkinter state changes are
    deferred to the main thread via ``root.after(0, ...)``.

    While the user is speaking, the sprite window switches to the open-mouth
    variant of the most recently predicted emotion. When the user stops
    speaking, the mouth is closed again and the "Speaking..." indicator is
    cleared.

    Args:
        is_talking : (bool)
            ``True`` when the VAD loop has detected that the
            user's microphone volume exceeds the configured
            loudness threshold; ``False`` once silence is
            detected.
    """
    root.after(0, lambda: talking.set("Speaking..." if is_talking else ""))
    # Toggle mouth open/closed on the sprite for the last known emotion.
    root.after(0, lambda: change_sprite(prev_emotion, is_talking))


# ── Background audio detection thread ────────────────────────────────────────
# stop_event is used to signal the background thread to exit cleanly when the
# main window closes. The thread is marked as a daemon so it will not prevent
# Python from exiting if mainloop() returns unexpectedly.
stop_event = threading.Event()
backend_thread = threading.Thread(
    target=run_emotion_detection,
    args=(
        on_result,
        on_mic_update,
        stop_event,
        loudness_threshold_slider,
        silence_threshold_slider,
        neutral_threshold_slider,
    ),
    daemon=True,
)
backend_thread.start()

# Block here until the user closes the settings window.
root.mainloop()

# Signal the background thread to finish its current iteration and exit.
stop_event.set()
