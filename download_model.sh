#!/bin/bash -x

# use wget -c to continue downloading if interrupted

wget -c --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget -c --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ" -O checkpoint.pt && rm -rf /tmp/cookies.txt
