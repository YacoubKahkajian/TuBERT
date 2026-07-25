"""
This module implements integration with Veadotube, a more technical
sequel to a popular piece of dedicated PNGTubing software, Veadotube
Mini, that's currently in early access.

I'm using the WebSocket-based API that Veadotube uses to read
thresholds set within the app and send the model's preditions to it.
This client is designed for non-Mini. I may make a version for Mini in
the future, though it would be more difficult since Mini doesn't allow for
custom thresholds AFAICT.

You can learn more about both here: https://veado.tube/
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from torch import full

from predict_emotion import setup_models_and_stream, transcribe_and_predict


def _default_instances_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".veadotube"))
    return base / "instances"


INSTANCES_DIR = _default_instances_dir()
INSTANCE_STALE_SECONDS = 10  # ignore instances whose timestamp is older than this


def discover_server() -> str:
    """
    Return the IP address of thefirst live veadotube server
    found in the instances folder.
    """
    if not INSTANCES_DIR.exists():
        raise RuntimeError(f"Instances folder not found: {INSTANCES_DIR}\n")

    now = time.time()
    candidates: list[tuple[int, str]] = []  # (timestamp, server)

    for entry in INSTANCES_DIR.iterdir():
        if not entry.is_file():
            continue
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
            ts: int = data["time"]
            server: str = data["server"]
        except Exception:
            continue

        if now - ts > INSTANCE_STALE_SECONDS:
            continue  # stale instance

        candidates.append((ts, server))

    if not candidates:
        raise RuntimeError("No live veadotube instances found.\n")

    # Pick the earliest-launched instance (lowest timestamp).
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _encode(payload: dict[str, Any]) -> str:
    """Wrap a payload dict in the `nodes: {...}` wire format."""
    return f"nodes: {json.dumps(payload)}"


def _decode(raw: str) -> dict[str, Any] | None:
    """Strip the channel prefix and parse JSON; returns None on failure."""
    prefix = "nodes:"
    if not raw.startswith(prefix):
        return None
    try:
        return json.loads(raw[len(prefix) :])
    except json.JSONDecodeError:
        return None


def _ws_uri(server: str) -> str:
    """Return the server IP as a WebSocket URL"""
    return f"ws://{server}?n=TuBERT"


def _state_payload(node_id: str, inner: dict[str, Any], type: str) -> dict[str, Any]:
    """
    Wraps a WS request to Veadotube in the entire required payload.

    Args:
        node_id : (str)
            ID of the node that is being viewed or changed.

        inner : (dict[str, Any])
            Usually of the format {"event": ..., "state": ...},
            where the value of "event" describes what to do
            with the node and the value of "state" contains
            the value to set the node to, if it is being edited.

        type : (str)
            Datatype of the node being selected.
    """
    return {
        "event": "payload",
        "type": type,
        "id": node_id,
        "payload": inner,
    }


async def _send_request(
    server: str,
    msg: str,
    callback=None,
) -> dict[str, Any] | None:
    """
    Send a WebSocket request to the currently running Veadotube instance.

    Args:
        server : (str)
            IP of the WS server where the Veadotube instance is running.

        msg : (str)
            Message to send to the WS server.

        callback : (dict[str, Any])
            Optional function to run on response.
    """
    async with websockets.connect(_ws_uri(server)) as ws:
        await ws.send(msg)
        await ws.recv()  # ACK frame, read and discard
        if callback is None:
            raw = await ws.recv()
            return _decode(str(raw))
        else:
            async for raw in ws:
                data = _decode(str(raw))
                callback(data)
            return None


async def _state_get(server: str, node_id: str) -> dict[str, Any] | None:
    """
    Retrieve the current value of a state WebSocket node with ID `node_id`

    Args:
        server : (str)
            IP of the WS server where the Veadotube instance is running.

        node_id : (str)
            ID of the node to retrieve the value of. Can be found in
            Veadotube by clicking on the node.

    Returns:
        string value with the name of the node's current state.
    """
    msg = _encode(_state_payload(node_id, {"event": "peek"}, "stateEvents"))
    data = await _send_request(server, msg)
    return data


async def _int_get(
    server: str,
    node_id: str,
) -> int:
    """
    Retrieve the current value of a number WebSocket node with ID `node_id`

    Args:
        server : (str)
            IP of the WS server where the Veadotube instance is running.

        node_id : (str)
            ID of the node to retrieve the value of. Can be found in
            Veadotube by clicking on the node.

    Returns:
        int value of the node.
    """
    msg = _encode(_state_payload(node_id, {"event": "get"}, "number"))
    data = await _send_request(server, msg)
    return data["payload"]["value"]


async def _bool_get(
    server: str,
    node_id: str,
    callback=None,
) -> bool | None:
    """
    Retrieve the current value of a boolean WebSocket node with ID `node_id`

    Args:
        server : (str)
            IP of the WS server where the Veadotube instance is running.

        node_id : (str)
            ID of the node to retrieve the value of. Can be found in
            Veadotube by clicking on the node.

        callback : (dict[str, Any])
            Optional function to run on response.

    Returns:
        bool value of the node.
    """
    msg = _encode(_state_payload(node_id, {"event": "listen"}, "boolean"))
    data = await _send_request(server, msg, callback)
    return data["payload"]["payload"]


async def _state_set(
    server: str,
    node_id: str,
    value: str,
) -> None:
    """
    Set the value of a state WebSocket node with ID `node_id` to string `value`

    Args:
        server : (str)
            IP of the WS server where the Veadotube instance is running.

        node_id : (str)
            ID of the node to retrieve the value of. Can be found in
            Veadotube by clicking on the node.

        value : (str)
            Value to set the node to.
    """
    msg = _encode(
        _state_payload(node_id, {"event": "set", "state": value}, "stateEvents")
    )
    async with websockets.connect(_ws_uri(server)) as ws:
        await ws.send(msg)


async def _int_set(
    server: str,
    node_id: str,
    value: int,
) -> None:
    """
    Set the value of a state WebSocket node with ID `node_id` to int `value`

    Args:
        server : (str)
            IP of the WS server where the Veadotube instance is running.

        node_id : (str)
            ID of the node to retrieve the value of. Can be found in
            Veadotube by clicking on the node.

        value : (int)
            Value to set the node to.
    """
    msg = _encode(_state_payload(node_id, {"event": "set", "value": value}, "number"))
    async with websockets.connect(_ws_uri(server)) as ws:
        await ws.send(msg)


server = discover_server()


async def main():
    is_speaking = False # This value is retrieved from the VAD pipeline in Veadotube.
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()
    predictor, vosk_model, audio_stream, sample_rate, samples_per_chunk = (
        setup_models_and_stream()
    )

    def on_bool(value):
        nonlocal is_speaking
        # Even though I have _bool_get return the specific boolean payload
        # from the response it sometimes still returns the entire object
        # anyways, but forcing the main loop to extract the payload again
        # from whatever is returned from _bool_get results in more
        # consistent behavior. Dunno why.
        is_speaking = value["payload"]

    def on_result(predicted_emotion, probs, predicted_confidence, transcript_text):
        """Send model's outputs to Veadotube"""
        print(transcript_text)
        asyncio.run_coroutine_threadsafe(
            _state_set(server, "ceb474aa", predicted_emotion), loop
        )
        asyncio.run_coroutine_threadsafe(
            _int_set(server, "50b092a5", int(predicted_confidence * 100)), loop
        )
        asyncio.run_coroutine_threadsafe(
            _state_set(server, "9c3c8d96", probs[1][0]), loop
        )

    bool_task = asyncio.create_task(_bool_get(server, "f63d468f", callback=on_bool))
    full_audio_chunk = []

    while not stop_event.is_set():
        await asyncio.sleep(0)  # yield to event loop so bool_task can run

        # While Veadotube takes audio input to handle VAD, this client program is
        # the one that actually stores what is being recorded with pyaudio. Once
        # Veadotube detects that the user is done speaking, the recorded audio is
        # then sent to the TuBERT model.
        if is_speaking:
            audio_chunk = audio_stream.read(
                samples_per_chunk, exception_on_overflow=False
            )
            np_audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)
            full_audio_chunk.append(np_audio_chunk)
        else:
            if len(full_audio_chunk) > 0:
                np_full_audio_chunk = np.array(full_audio_chunk).flatten()
                full_audio_chunk = []
                emotion, probs, confidence, transcript_text = transcribe_and_predict(
                    predictor, vosk_model, np_full_audio_chunk
                )
                on_result(emotion, probs, confidence, transcript_text)

    try:
        await bool_task
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()


asyncio.run(main())
