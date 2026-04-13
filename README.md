# TuBERT: Multimodal Speech Emotion Recognition For Real-Time Avatar Control

*This project was developed for my senior thesis at Princeton University. The paper will be published soon.*

![Screenshot of the TuBERT GUI](/gui.jpg)

## About
TuBERT is a multimodal speech emotion recognition model that runs in real-time and on-device. I designed it with PNGTubers in mind, but there are plenty of other applications for it as well!

This repository contains the code to try a TuBERT model, such as the one I used for my thesis, using a (very) basic GUI. If this project ever gets more attention I will consider adding more features that make the GUI fit for an actual livestream.

This repository also includes the code I used to train ([instructions](./model/setup/README.md)) and test ([instructions](./model/stats/README.md)) the model.

## Installation
If this project gets more attention I will also consider making a packaged version of TuBERT and the GUI you can install as an application with one click. Installation should be pretty painless either way, through.

1. Clone the repository: `git clone https://github.com/YacoubKahkajian/TuBERT.git`
2. Open the cloned folder: `cd TuBERT`
3. Download a pre-trained TuBERT model from HuggingFace and place it in `/model`. You can either [download the model here](https://huggingface.co/yacoubk/TuBERT/resolve/main/tubert.pt) and drag the file into the folder or use the command line for this: `curl -L https://huggingface.co/yacoubk/TuBERT/resolve/main/tubert.pt --output ./model
`
4. `python -m venv .venv && source .venv/bin/activate`
5. `pip install -e .`

## Usage
After you have followed the instructions above to install TuBERT, run the command `python main.py` inside the repo directory to open the TuBERT GUI. 

I have only tested the TuBERT model on Mac and Linux and the TuBERT GUI on Mac. I can't think of a reason TuBERT shouldn't be able to run on Windows, but let me know if there's any compatibility issues you run into regardless.
