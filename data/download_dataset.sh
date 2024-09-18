#!/bin/bash -x

# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RzqSbSnbMEwLUagV0GThfpm9JJXePGkV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RzqSbSnbMEwLUagV0GThfpm9JJXePGkV" -O edge_aistpp.zip && rm -rf /tmp/cookies.txt
# gdown https://docs.google.com/uc?export=download&id=1RzqSbSnbMEwLUagV0GThfpm9JJXePGkV

# manually download edge_aistpp.zip via
# https://drive.usercontent.google.com/download?id=1RzqSbSnbMEwLUagV0GThfpm9JJXePGkV&export=download&authuser=0

unzip edge_aistpp.zip
