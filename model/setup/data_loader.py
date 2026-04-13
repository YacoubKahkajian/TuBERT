"""
Parses MELD and preprocessed data as
Pytorch datasets
"""

import json
from pathlib import Path

import numpy as np
import torch
from pandas import read_csv
from torch.utils.data import DataLoader, Dataset, Subset

import config

# Emotion mapping shared between IEMOCAPFineTuneDataset and test.py.
# "other" consensus labels are excluded (no valid mapping).
IEMOCAP_TO_MODEL_EMOTION = {
    "neutral": "neutral",
    "happiness": "joy",
    "sadness": "sadness",
    "anger": "anger",
}


class MELDDataset(Dataset):
    """
    MELD Dataset, before preprocessing.
    Returns paths of audio files when indexed.
    """

    def __init__(self, split="train", meld_dir=config.MELD_ROOT):
        """
        Args:
            split : (str)
                The dataset split to load: 'train', 'dev', or 'test'.
            meld_dir : (Path)
                Root directory of the MELD dataset containing CSV files
                and the audio subdirectory.
        """
        self.data = read_csv(f"{meld_dir}/{split}_sent_emo.csv")
        self.split = split
        self.meld_dir = meld_dir

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        """
        Return the audio file path and metadata for a single MELD utterance.

        Args:
            idx : (int)
                Row index into the split CSV.

        Returns:
            path : (str)
                Path to the WAV file, e.g.
                ``{meld_dir}/audio/train/dia3_utt1.wav``.
            metadata : (dict)
                Dictionary with keys ``speaker``, ``emotion``, ``sentiment``,
                ``dialogue_id``, ``utterance_id``, ``transcript``, and
                ``series_number``.
        """
        sample = self.data.iloc[idx]
        dia_id = sample["Dialogue_ID"]
        utt_id = sample["Utterance_ID"]
        path = f"{self.meld_dir}/audio/{self.split}/dia{dia_id}_utt{utt_id}.wav"
        metadata = {
            "speaker": sample["Speaker"],
            "emotion": sample["Emotion"],
            "sentiment": sample["Sentiment"],
            "dialogue_id": int(sample["Dialogue_ID"]),
            "utterance_id": int(sample["Utterance_ID"]),
            "transcript": sample["Utterance"],
            "series_number": int(sample["Sr No."]),
        }
        return path, metadata


class PreprocessedDataset(Dataset):
    """
    Returns tensors of preprocessed data when indexed.
    Expects the directory layout produced by ``preprocess_data.py``::

        {preprocessed_dir}/
            {split}_metadata.json
            {split}/
                0_audio.npy
                0_text.npy
                1_audio.npy
                ...

    ``"disgust"`` labels are remapped to ``"anger"`` and ``"fear"`` labels
    are remapped to ``"sadness"`` to reduce the number of low-frequency classes.
    """

    def __init__(self, split="train", preprocessed_dir=config.PREPROCESSED_ROOT):
        """
        Args:
            split : (str)
                The dataset split to load: 'train', 'dev', or 'test'.
            preprocessed_dir : (Path)
                Root directory containing the preprocessed ``.npy`` feature
                files and JSON metadata produced by ``preprocess_data.py``.
        """
        with open(f"{preprocessed_dir}/{split}_metadata.json", "r") as f:
            self.data = json.load(f)
        self.split = split
        self.preprocessed_dir = preprocessed_dir

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        """
        Load pre-extracted features and the integer emotion label for one sample.
        ``"disgust"`` is remapped to ``"anger"``and ``"fear"`` is remapped to
        ``"sadness"`` before the lookup.

        Args:
            idx : (int)
                Sample index; corresponds to the numeric prefix of the saved
                ``.npy`` files (e.g. ``idx=3`` → ``3_audio.npy``).

        Returns:
            audio_features : (torch.Tensor)
                Shape ``(time_steps, audio_feature_dim)``. CMVN-normalised
                MFCC+delta+delta-delta sequence.
            text_embedding : (torch.Tensor)
                Shape ``(text_embedding_dim,)``. DistilBERT [CLS]-token
                embedding for the utterance transcript.
            label : (torch.LongTensor)
                Shape ``(1,)``. Integer emotion class index in the range
                ``[0, len(config.EMOTIONS) - 1]``.
        """
        # Get sample based on series number in spreadsheet
        sample = self.data[idx]
        audio_features_path = config.PREPROCESSED_ROOT / self.split / f"{idx}_audio.npy"
        text_features_path = config.PREPROCESSED_ROOT / self.split / f"{idx}_text.npy"
        audio_features = torch.from_numpy(np.load(audio_features_path))
        text_features = torch.from_numpy(np.load(text_features_path))

        # Convert label to int (scikit-learn doesn't accept it otherwise)
        emotion = sample["emotion"]
        if emotion == "disgust":
            emotion = "anger"
        if emotion == "fear":
            emotion = "sadness"
        label = config.EMOTION_TO_IDX[emotion]
        label = torch.LongTensor([label])
        return audio_features, text_features, label


class IEMOCAPDataset(Dataset):
    """
    IEMOCAP dataset, using the lite version here:
    https://www.kaggle.com/datasets/sangayb/iemocap

    In my paper, only used for TESTING the final model. Reads
    ``iemocap_metadata.json`` produced by ``iemocap_create_metadata.py``
    and maps each entry to its WAV file path. Does not extract
    features itself.
    """

    def __init__(self, iemocap_dir=config.IEMOCAP_ROOT):
        with open(f"{iemocap_dir}/iemocap_metadata.json", "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        session_numb = sample["session_number"]
        file_name = sample["file_name"]
        path = (
            config.IEMOCAP_ROOT
            / f"Session{session_numb}"
            / "sentences"
            / "wav"
            / file_name[:-5]
            / f"{file_name}.wav"
        )
        return path, sample


class IEMOCAPFineTuneDataset(Dataset):
    """
    IEMOCAP dataset for fine-tuning, with on-the-fly feature extraction
    (as opposed to how MELD is preprocessed).

    Filters to a specified set of sessions and excludes samples whose
    consensus_emotion is "other" (no valid mapping to the model's emotion
    set).

    Args:
        sessions : (list[int])
            Session numbers to include, e.g. ``[1, 2, 3, 4]``.
        audio_extractor : (AudioFeatureExtractor)
            Shared extractor instance to avoid reloading torchaudio transforms.
        text_extractor : (TextFeatureExtractor)
            Shared extractor instance to avoid reloading DistilBERT.
    """

    def __init__(self, sessions, audio_extractor, text_extractor):
        with open(config.IEMOCAP_ROOT / "iemocap_metadata.json", "r") as f:
            raw = json.load(f)

        # Filter to the requested sessions and mappable emotions only
        self.data = [
            s
            for s in raw
            if s["session_number"] in sessions
            and s["consensus_emotion"] in IEMOCAP_TO_MODEL_EMOTION
        ]

        self.audio_extractor = audio_extractor
        self.text_extractor = text_extractor

        print(
            f"IEMOCAPFineTuneDataset: {len(self.data)} samples from sessions {sessions}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        session_numb = sample["session_number"]
        file_name = sample["file_name"]

        audio_path = str(
            config.IEMOCAP_ROOT
            / f"Session{session_numb}"
            / "sentences"
            / "wav"
            / file_name[:-5]
            / f"{file_name}.wav"
        )

        # On-the-fly feature extraction
        audio_features = torch.FloatTensor(self.audio_extractor.extract(audio_path))
        text_embedding = torch.FloatTensor(
            self.text_extractor.extract(sample["transcription"])
        )

        # Map IEMOCAP consensus label to the model's emotion set
        emotion = IEMOCAP_TO_MODEL_EMOTION[sample["consensus_emotion"]]
        label = torch.LongTensor([config.EMOTION_TO_IDX[emotion]])

        return audio_features, text_embedding, label


def collate_fn(batch):
    """
    Pad variable-length audio sequences to the longest sequence in a batch.

    Each element of ``batch`` is a ``(audio_tensor, text_tensor, label_tensor)``
    tuple as returned by PreprocessedDataset or IEMOCAPFineTuneDataset. Audio
    tensors may have different numbers of time-steps; shorter ones are
    zero-padded to ``max(lengths)`` so they can be stacked into a single tensor.

    A ``lengths`` tensor recording the true (unpadded) length of each sequence
    is returned alongside the padded batch so that MultiModalEmotionRNN can
    pass it to ``pack_padded_sequence`` and AttentionLayer.

    Args:
        batch : (list[tuple])
            List of ``(audio, text, label)`` tuples where:
              - ``audio`` is a ``(time_steps, audio_dim)`` float tensor,
              - ``text`` is a ``(text_dim,)`` float tensor,
              - ``label`` is a ``(1,)`` long tensor.

    Returns:
        audio_batch : (torch.Tensor)
            Shape ``(batch_size, max_len, audio_dim)``. Zero-padded audio sequences.
        text_batch : (torch.Tensor)
            Shape ``(batch_size, text_dim)``. Stacked text embeddings.
        labels_batch : (torch.Tensor)
            Shape ``(batch_size,)``. Integer class labels.
        lengths_batch : (torch.LongTensor)
            Shape ``(batch_size,)``. True (unpadded) audio sequence length
            for each sample.
    """
    audio_features, text_embeddings, labels = zip(*batch)

    # Get max sequence length in batch
    max_len = max(x.shape[0] for x in audio_features)

    # Pad audio sequences
    padded_audio = []
    lengths = []
    for audio in audio_features:
        length = audio.shape[0]
        lengths.append(length)

        # Pad if necessary
        if length < max_len:
            padding = torch.zeros(max_len - length, audio.shape[1])
            padded = torch.cat([audio, padding], dim=0)
        else:
            padded = audio

        padded_audio.append(padded)

    # Stack into batch tensors
    audio_batch = torch.stack(padded_audio)  # (batch, max_len, audio_dim)
    text_batch = torch.stack(text_embeddings)  # (batch, text_dim)
    labels_batch = torch.cat(labels)  # (batch,)
    lengths_batch = torch.LongTensor(lengths)  # (batch,)

    return audio_batch, text_batch, labels_batch, lengths_batch


def get_dataloader(
    dataset="preprocessed",
    split="train",
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    balance_neutral=True,
    neutral_ratio=2.5,
):
    """
    Build a DataLoader for MELD data (raw or preprocessed).

    When ``balance_neutral=True`` and ``split="train"``, the neutral class is
    randomly undersampled so that its size is at most ``neutral_ratio`` times
    the average count of the other emotion classes.

    Args:
        dataset : (str)
            ``"preprocessed"`` to use PreprocessedDataset (default), or
            ``"meld"`` to use MELDDataset.
        split : (str)
            Dataset split to load: ``"train"``, ``"dev"``, or ``"test"``.
        batch_size : (int)
            Number of samples per batch. Defaults to ``config.BATCH_SIZE``.
        shuffle : (bool)
            Whether to shuffle the dataset each epoch. Defaults to ``True``.
        num_workers : (int)
            Number of worker processes for data loading. Defaults to ``4``.
        balance_neutral : (bool)
            Whether to undersample the neutral class in the training split.
            Defaults to ``True``.
        neutral_ratio : (float)
            Maximum ratio of neutral samples to the average count of other
            emotion classes after undersampling. Defaults to ``2.5``.

    Returns:
        loader : (torch.utils.data.DataLoader)
            Configured DataLoader yielding
            ``(audio_batch, text_batch, labels_batch, lengths_batch)`` tuples
            via ``collate_fn``.
    """
    dataset = (
        MELDDataset(split=split)
        if dataset == "meld"
        else PreprocessedDataset(split=split)
    )

    if balance_neutral and split == "train":
        print("Balancing neutral class...")
        neutral_idx = config.EMOTION_TO_IDX["neutral"]

        # Separate neutral and non-neutral indices
        neutral_indices = []
        emotion_indices = []

        for i, (_, _, label) in enumerate(dataset):
            if label.item() == neutral_idx:
                neutral_indices.append(i)
            else:
                emotion_indices.append(i)

        print(
            f"Original: {len(neutral_indices)} neutral, {len(emotion_indices)} emotion"
        )

        # Calculate target number of neutral samples
        avg_emotion_count = len(emotion_indices) / (len(config.EMOTIONS) - 1)
        target_neutral = int(avg_emotion_count * neutral_ratio)

        np.random.seed(42)
        sampled_neutral = np.random.choice(
            neutral_indices,
            size=min(target_neutral, len(neutral_indices)),
            replace=False,
        ).tolist()

        # Combine and create subset
        idx_to_keep = emotion_indices + sampled_neutral
        print(
            f"After balancing: {len(sampled_neutral)} neutral, {len(emotion_indices)} emotion"
        )

        dataset = Subset(dataset, idx_to_keep)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(torch.device == "cuda"),
    )


def get_iemocap_dataloaders(batch_size=config.BATCH_SIZE):
    """
    Build train and validation DataLoaders for IEMOCAP fine-tuning.
    Trains on sessions 1–4 and validates on session 5.

    Args:
        batch_size : (int)
            Batch size for both the train and validation loaders.
            Defaults to ``config.BATCH_SIZE``.

    Returns:
        train_loader : (torch.utils.data.DataLoader)
            DataLoader over sessions 1–4, shuffled.
        val_loader : (torch.utils.data.DataLoader)
            DataLoader over session 5, not shuffled.
    """
    from feature_extractor import AudioFeatureExtractor, TextFeatureExtractor

    print("Loading feature extractors for IEMOCAP fine-tuning...")
    audio_extractor = AudioFeatureExtractor()
    text_extractor = TextFeatureExtractor(device=config.DEVICE)

    train_dataset = IEMOCAPFineTuneDataset(
        sessions=[1, 2, 3, 4],
        audio_extractor=audio_extractor,
        text_extractor=text_extractor,
    )
    val_dataset = IEMOCAPFineTuneDataset(
        sessions=[5],
        audio_extractor=audio_extractor,
        text_extractor=text_extractor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # must be 0 — DistilBERT cannot be shared across workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return train_loader, val_loader
