#!/bin/bash

python3 -m pip install parlai

python3 -m parlai.scripts.display_data --task dailydialog --datapath dialogue
python3 -m parlai.scripts.display_data --task blended_skill_talk --datapath dialogue
python3 -m parlai.scripts.display_data --task empathetic_dialogues --datapath dialogue
python3 -m parlai.scripts.display_data --task personachat --datapath dialogue