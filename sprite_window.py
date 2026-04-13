"""
This module creates and manages a floating Tkinter ``Toplevel`` window that
shows an animated character sprite whose appearance reflects the most recently
predicted emotion and whether the user is currently speaking.

Sprites are image files stored in the ``sprites/`` directory and are named
with the convention::

    <emotion>_<mouth_state>.<ext>

For example: ``joy_open.jpg``, ``neutral_closed.png``.

When a matching sprite file cannot be found, the window falls back to
rendering the emotion label and mouth-state as centred text on the canvas.

Public API
----------
``sprite_window(root)``: creates the window and returns a ``change_sprite``
callable that the caller uses to update the displayed sprite at any time.
"""

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk


def sprite_window(root):
    """
    Create the TuBERT sprite display window and return its update function.

    Builds a fixed-size (300x300) dark-background ``Toplevel`` window
    containing a single ``Canvas``.  The canvas is redrawn every time the
    returned ``change_sprite`` callable is invoked.

    Args:
        root : (tk.Tk)
            The parent Tk root window.  The new ``Toplevel`` is
            attached to ``root`` so that it is destroyed automatically when
            the root window closes.

    Returns:
        change_sprite : (callable)
            A function with the signature
            ``change_sprite(emotion: str, mouth_open: bool) -> None``
            that updates the displayed sprite to match the given emotion
            and mouth state.
    """
    window = tk.Toplevel(root)
    window.title("TuBERT")
    window.resizable(False, False)

    # Tkinter variables that track the current display state.
    curr_emotion = tk.StringVar()
    curr_mouth_open = tk.BooleanVar()

    # Debug labels (disabled; uncomment to show state values on the canvas).
    # tk.Label(window, textvariable=curr_emotion).grid(column=0, row=0, sticky="w")
    # tk.Label(window, textvariable=curr_mouth_open).grid(column=0, row=1, sticky="w")

    canvas = tk.Canvas(
        window, width=300, height=300, bg="#1e1e1e", highlightthickness=0
    )
    canvas.pack(fill="both", expand=True)

    def _render():
        """
        Redraw the canvas to reflect the current emotion and mouth state.

        Looks up the appropriate sprite image from the ``sprites/`` directory
        using a glob search for a file whose name matches
        ``<emotion>_<open|closed>.*``.  The image is scaled to fit the
        canvas while preserving its aspect ratio (via ``Image.thumbnail``),
        then drawn centred on the canvas.

        If no matching file exists, or if the file cannot be opened, the
        function falls back to rendering the emotion name and mouth-state
        boolean as centred white text on the dark canvas background.

        """
        mouth_key = "open" if curr_mouth_open.get() else "closed"
        path = next(Path("sprites").rglob(f"{curr_emotion.get()}_{mouth_key}.*"), None)
        canvas.delete("all")

        if path:
            try:
                img = Image.open(path)
                canvas_w = canvas.winfo_width() or 300
                canvas_h = canvas.winfo_height() or 300
                img.thumbnail((canvas_w, canvas_h))
                photo = ImageTk.PhotoImage(img)
                canvas._photo_ref = photo  # prevent garbage collection
                x = canvas_w // 2
                y = canvas_h // 2
                canvas.create_image(x, y, anchor="center", image=photo)
                return
            except Exception:
                pass

            # Fallback: render text labels when the image cannot be loaded.
            canvas_w = canvas.winfo_width() or 300
            canvas_h = canvas.winfo_height() or 300
            canvas.create_text(
                canvas_w // 2,
                canvas_h // 2 - 12,
                text=curr_emotion.get(),
                fill="white",
                font=("SF Pro", 22, "bold"),
            )
            canvas.create_text(
                canvas_w // 2,
                canvas_h // 2 + 18,
                text=str(curr_mouth_open.get()),
                fill="#888888",
                font=("SF Pro", 13),
            )

    def change_sprite(emotion, mouth_open):
        """
        Update the displayed sprite to match a new emotion and mouth state.

        Args:
            emotion : (str)
                The emotion to display.  Must correspond to a filename
                stem present in the ``sprites/`` directory for an image to
                be shown; otherwise the text fallback is used.
            mouth_open : (bool)
                ``True`` if the character's mouth should be shown open
                (i.e. the user is currently speaking), ``False``
                for the closed-mouth idle pose.
        """
        curr_emotion.set(emotion)
        curr_mouth_open.set(mouth_open)
        _render()

    return change_sprite
