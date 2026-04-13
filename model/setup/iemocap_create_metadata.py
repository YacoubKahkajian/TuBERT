"""
The original IEMOCAP dataset has a bunch of data
we don't need for this project, and even lighter forks
such as https://www.kaggle.com/datasets/sangayb/iemocap
keep the same file structure.

The scripts in this file create a single JSON file, similar
to the CSV files that come with MELD, that list every single
snippet in IEMOCAP that comply to the requirements described
in the paper, their transcript, and their emotion label.

"""

import json
import re
from pathlib import Path

import config

# --- Emotion mappings ------------------------------
# Short codes used in the header line of label files
LABEL_CODE_TO_EMOTION = {
    "neu": "neutral",
    "hap": "happiness",
    "sad": "sadness",
    "ang": "anger",
    "sur": "surprise",
    "fru": "frustration",
    "exc": "excited",
    "fea": "fear",
    "dis": "disgust",
    "oth": "other",
    "xxx": "unknown",
}

# Full emotion names used in C- annotator lines
ANNOTATOR_LABEL_TO_EMOTION = {
    "neutral": "neutral",
    "happiness": "happiness",
    "sad": "sadness",
    "sadness": "sadness",
    "angry": "anger",
    "anger": "anger",
    "excited": "excited",
    "frustration": "frustration",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "other": "other",
}

# For the IEMOCAP eval (standard in literature):
# excited → happiness, everything else dropped if not in set
FIVE_CLASS_MAP = {
    "neutral": "neutral",
    "happiness": "happiness",
    "excited": "happiness",  # standard mapping per past literature
    "sadness": "sadness",
    "anger": "anger",
    "frustration": "anger",
    "surprise": "surprise",
}


def get_transcripts():
    """
    Read improvisation-session transcripts for all 5 IEMOCAP sessions.
    Searches each session directory for transcript .txt files whose name
    contains "impro" (improvised dialogues, as opposed to scripted ones).
    The utterance file name (first token) and transcription text (after
    "]: ") are extracted via a regular expression and stored together.

    Returns:
        dict_transcriptions : (dict[str, str])
            Mapping from utterance file name
            (e.g. "Ses01F_impro01_F000") to its transcription string
            (e.g. "Hello there."). If a file name appears more than
            once across sessions it is overwritten by the last occurrence,
            though in practice all file names are unique.
    """
    file_names = []
    transcriptions = []
    for session_numb in range(1, 6):
        session_dir = config.IEMOCAP_ROOT / f"Session{session_numb}"
        for file in session_dir.glob("**/transcriptions/*impro*.txt"):
            session_name = file.stem
            with open(
                session_dir / "dialog" / "transcriptions" / f"{session_name}.txt", "r"
            ) as f:
                for line in f:
                    columns = re.match(r"(\S+) \[(\S+)\]: (.+)", line)
                    if columns:
                        file_names.append(columns.group(1))
                        transcriptions.append(columns.group(3))
    dict_transcriptions = dict(zip(file_names, transcriptions))
    return dict_transcriptions


def get_sentiments():
    """
    Iterates over all 5 IEMOCAP sessions and reads the EmoEvaluation
    .txt files for improvised dialogues. Each file contains three types
    of lines that are parsed by distinct regular expressions:

    * Header lines: one per utterance, containing the file name and the
      original consensus label code (e.g. "neu", "hap"). These seed
      file_names, labelled_emotions, and session_numbs.

    * Annotator lines (C): one per human annotator, containing
      a full emotion label string (e.g. "neutral", "happiness").
      Each label is mapped through FIVE_CLASS_MAP (which folds
      "frustrated" -> "anger" and "excited" -> "happiness", etc.) and
      counted. Labels with no mapping increment the "other" bucket.

    * Valence score lines (A): one per utterance, appearing after
      all annotator lines for that utterance. On the first such line the
      annotator counts are resolved: if any emotion has >= 2 votes it
      becomes the consensus; otherwise the consensus is "other". The
      counter is then reset for the next utterance.

    Returns:
        file_names : (list[str])
            Utterance file names in parse order.
        labelled_emotions : (list[str])
            Original consensus label decoded from the header line's short
            code via LABEL_CODE_TO_EMOTION.
        consensus_emotions : (list[str])
            Majority-vote emotion derived from annotator lines. "other"
            when no majority is reached or the top emotion is not in the
            five-class set.
        session_numbs : (list[int])
            IEMOCAP session number (1–5) for each utterance.
    """
    file_names = []
    labelled_emotions = []
    consensus_emotions = []
    session_numbs = []
    for session_numb in range(1, 6):
        session_dir = config.IEMOCAP_ROOT / f"Session{session_numb}"
        for file in session_dir.glob("**/EmoEvaluation/*impro*.txt"):
            session_name = file.stem
            consensus_counter_reset = False
            consensus_counter = {
                "neutral": 0,
                "happiness": 0,
                "sadness": 0,
                "anger": 0,
                "surprise": 0,
                "other": 0,
            }
            with open(
                session_dir / "dialog" / "EmoEvaluation" / f"{session_name}.txt", "r"
            ) as f:
                for line in f:
                    header_line = re.match(
                        r"(\S+\s[-]\s\S+)\s+(\S+)\s+(\S+)\s([[].+[]])", line
                    )
                    annotator_label = re.match(r"[C][-].+[:]\s(\S+)[;]\s[(].*[)]", line)
                    valence_score = re.match(r"[A][-].+", line)
                    if header_line:
                        consensus_counter_reset = False
                        file_names.append(header_line.group(2))
                        session_numbs.append(session_numb)
                        labelled_emotion = header_line.group(3)
                        labelled_emotions.append(
                            LABEL_CODE_TO_EMOTION[labelled_emotion]
                        )
                    elif annotator_label:
                        consensus_emotion = annotator_label.group(1).lower()
                        if consensus_emotion not in FIVE_CLASS_MAP:
                            consensus_counter["other"] += 1
                        else:
                            consensus_emotion = FIVE_CLASS_MAP[consensus_emotion]
                            consensus_counter[consensus_emotion] += 1
                        # print(consensus_counter)
                    elif valence_score and not consensus_counter_reset:
                        consensus_counter_reset = True
                        if max(consensus_counter.values()) >= 2:
                            consensus_emotion = max(
                                consensus_counter, key=consensus_counter.get
                            )
                        else:
                            consensus_emotion = "other"
                        consensus_counter = {
                            "neutral": 0,
                            "happiness": 0,
                            "sadness": 0,
                            "anger": 0,
                            "surprise": 0,
                            "other": 0,
                        }
                        consensus_emotions.append(consensus_emotion)

                    else:
                        continue
    return file_names, labelled_emotions, consensus_emotions, session_numbs


def create_json():
    """
    Merge transcripts and sentiment data and write iemocap_metadata.json,
    a list of dicts, each containing:

    * "file_name":          utterance file name (e.g. "Ses01F_impro01_F000").
    * "labelled_emotion":   original header-line consensus label.
    * "consensus_emotion":  majority-vote label from annotator lines.
    * "transcription":      utterance text from the transcript file.
    * "session_number":     IEMOCAP session number (1–5).

    The resulting list is serialised to data/IEMOCAP/iemocap_metadata.json
    with 4-space indentation.
    """
    transcriptions = get_transcripts()
    file_names, labelled_emotions, consensus_emotions, session_numbs = get_sentiments()
    dict_list = []
    for i in range(len(file_names)):
        dict_list.append(
            {
                "file_name": file_names[i],
                "labelled_emotion": labelled_emotions[i],
                "consensus_emotion": consensus_emotions[i],
                "transcription": transcriptions[file_names[i]],
                "session_number": session_numbs[i],
            }
        )
    with open("data/IEMOCAP/iemocap_metadata.json", "w") as f:
        json.dump(dict_list, f, indent=4)


if __name__ == "__main__":
    create_json()
