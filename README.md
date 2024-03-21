# StarFox-AI
A project focused on computer vision and spatial awareness in AI. Operates on higan using a virtual controller, right now still in the stage of intial development, including the algorithms for simple training. (Works only in linux right now)

![FoxAI in training game mode](https://github.com/GuineaBot3Labs/StarFox-AI/blob/main/FoxAI.png)

Here, FoxAI plays a game of the "Training" game mode, right now it is currently in it's infant stage, like GuineaBot3 it did start very small, (which still has bugs by the way, thus we haven't seen it for a long time, help [here.](https://github.com/GuineaBot3Labs/deep-GuineaBot3-lichess-bot))

# Update!

This repository will probably use a [modified version of higan](https://github.com/GuineaBot3Labs/higan-env) to interface with, until then however, this project is post-poned.

## Setup (Documentation incomplete!) ##
First off, you will have to install higan:

    sudo apt install higan

or if your linux installation is anything like mine, then run:

    sudo nano /etc/apt/sources.list

where you will insert the lines for ubuntu's repo, and if the key is missing then add it, finally, run:

    sudo apt update
    sudo apt install higan

You will then have to legally obtain a StarFox ROM. Load this ROM into the emulator, and do the rest. (May require fiddling around with the settings.)

Download the requirements like so:

    python3 -m pip install -r requirements.txt

## Currently in development, come back later! (unless you're willing to help) ##

We need help with this project, please [fork](../../fork) this repository and implement some cool features and open a pull request! (i.e., a complete integration into higan's emulator or something.) Or if you want to be more directly involved, you can join [here.](https://github.com/GuineaBot3Labs/join)

We are also in need of players on higan playing this game, so we can train Peppy on the gameplay to evaluate FoxAI in real time. Please use the "higan.py" file to continously take screenshots of higan (note: This data will not be labeled, it will be labeled by our [Dataset Labeling](https://github.com/orgs/GuineaBot3Labs/teams/dataset-labelers) team. We need more people in that team, join [here and request a specific team with a label.](https://github.com/GuineaBot3Labs/join)
)

![FoxAI in training game mode](https://github.com/GuineaBot3Labs/StarFox-AI/blob/main/Vision.png)

# Disclaimer
This software does not condone pirating in any way. All credit other than the AI research to Nintendo and higan, which have both made this project possible.
