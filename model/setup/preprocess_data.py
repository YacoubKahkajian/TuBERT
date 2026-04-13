"""
Extracting MFCC+delta+delta-delta audio features and DistilBERT text
embeddings is computationally expensive. This script runs the
extraction once and saves the results to disk so that training reads cheap
NumPy arrays instead.

For each MELD split (``train``, ``dev``, ``test``):

1. Iterate over :class:`~data_loader.MELDDataset` to collect all WAV file
   paths and utterance transcripts whose audio files actually exist on disk.
2. Extract MFCC+delta+delta-delta features (with per-utterance CMVN
   normalisation) for every audio file using
   :class:`~feature_extractor.AudioFeatureExtractor`.
3. Extract DistilBERT [CLS]-token embeddings for every transcript using
   :class:`~feature_extractor.TextFeatureExtractor` (can be disabled via
   ``embed_text=False`` if DistilBERT is not available locally).
4. Save each utterance's audio features as
   ``{PREPROCESSED_ROOT}/{split}/{i}_audio.npy`` and its text embedding as
   ``{PREPROCESSED_ROOT}/{split}/{i}_text.npy``.
5. Write a ``{PREPROCESSED_ROOT}/{split}_metadata.json`` file containing
   the list of metadata dicts (speaker, emotion, dialogue/utterance IDs, etc.)
   in the same order as the saved ``.npy`` files.

The resulting directory layout is consumed by
:class:`~data_loader.PreprocessedDataset` during training.
"""

import json
import os
from pathlib import Path

import numpy as np
from data_loader import MELDDataset
from feature_extractor import AudioFeatureExtractor, TextFeatureExtractor
from tqdm import tqdm

import config


def preprocess(split="train", embed_text=True):
    """
    Extract and save audio + text features for one MELD split.

    Feature arrays are written to
    ``{config.PREPROCESSED_ROOT}/{split}/{i}_audio.npy`` and
    ``{config.PREPROCESSED_ROOT}/{split}/{i}_text.npy``, where ``i`` is the
    zero-based index into the list of found files.  A companion
    ``{config.PREPROCESSED_ROOT}/{split}_metadata.json`` is also written
    containing speaker, emotion, dialogue/utterance ID, and transcript for
    every saved sample.

    Args:
        split : (str)
            The MELD split to preprocess — ``"train"``, ``"dev"``,
            or ``"test"``.  Defaults to ``"train"``.
        embed_text : (bool)
            Whether to extract and save DistilBERT text embeddings.
            Set to ``False`` if the machine cannot run DistilBERT locally
            (a Colab notebook is provided as an alternative).
            Defaults to ``True``.
    """
    data = MELDDataset(split=split)

    os.makedirs(config.PREPROCESSED_ROOT / split, exist_ok=True)

    # Collect paths and metadata for every file that exists on disk
    file_paths = []
    transcript_texts = []
    all_metadata = []

    for idx in range(len(data)):
        audio_path, metadata = data[idx]
        if Path(audio_path).is_file():
            file_paths.append(audio_path)
            transcript_texts.append(metadata["transcript"])
            all_metadata.append(metadata)

    # Extract features
    print("Extracting audio features...")
    all_audio_features = AudioFeatureExtractor().extract_batch(file_paths)

    if embed_text:
        print("Extracting text features...")
        all_text_features = TextFeatureExtractor(device="cpu").extract_batch(
            transcript_texts
        )

    # Save features to disk
    for i in tqdm(
        range(len(file_paths)),
        desc=f"Saving features to {config.PREPROCESSED_ROOT / split}",
    ):
        np.save(
            config.PREPROCESSED_ROOT / split / f"{i}_audio.npy",
            all_audio_features[i],
        )
        if embed_text:
            np.save(
                config.PREPROCESSED_ROOT / split / f"{i}_text.npy",
                all_text_features[i],
            )

    # Save metadata alongside the features
    metadata_path = config.PREPROCESSED_ROOT / f"{split}_metadata.json"
    print(f"Saving feature metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f)


def main():
    """
    Preprocess all three MELD splits (train, dev, test).

    Creates ``config.PREPROCESSED_ROOT`` if it does not exist, then calls
    :func:`preprocess` for each of the three MELD splits in sequence.
    Run this script once before starting training to avoid repeated feature
    extraction during each epoch.
    """
    print("Preprocessing MELD dataset (CMVN normalisation enabled)...")
    config.PREPROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        preprocess(split)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
