#!/bin/bash -x
brew install libsndfile
echo "import os; os.environ['DYLD_LIBRARY_PATH'] = '$(brew info libsndfile | grep -A1 Installed | tail -1 | cut -d' ' -f1)/lib'" > $(python -c "import site; print(site.getsitepackages()[0])")/sitecustomize.py
cat $(python -c "import site; print(site.getsitepackages()[0])")/sitecustomize.py
