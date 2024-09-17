#!/bin/bash -x
pip install -r requirements.txt
# pytorch3d needs --no-build-isolation to find torch
# to resolve the error message of
# ModuleNotFoundError: No module named 'torch'
pip install -e git+https://github.com/lloydchang/facebookresearch-pytorch3d@lloydchang-patch-1#egg=pytorch3d --no-build-isolation
