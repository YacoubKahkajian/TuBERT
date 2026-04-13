# Run this script after you download the MELD data and placed
# it in the project folder.

# The MELD dataset uses MP4 files, but for our purposes, we
# only need the audio in these videos. This script extracts the
# audio from the MELD data using ffmpeg. Be sure the folder
# containing the data is named `MELD` and the folders containing
# the data are named `dev`, `test` and `train`.

splits=("dev" "test" "train")

for split in "${splits[@]}"; do
    mkdir -p MELD/audio/$split
    for file in MELD/$split/*.mp4; do
        ffmpeg -i "$file" -ar 16000 -ac 1 -y "MELD/audio/$split/$(basename "$file" .mp4).wav"
    done
done
