#!/bin/bash
# $1: repo URL
# $2: directory to instrument in repository
# $3: test directory in repository

REPO_NAME=$(echo $1 | grep -o '[^/]*$')

mkdir ./data
mkdir ./data/repos

# Download repo
git -C ./data/repos clone $1

# Install requirements
cd ./data/repos/$REPO_NAME
python3 setup.py install

# Install additional requirements
# Rich
pip install commonmark
pip install pygments
pip install attr
# Requests
pip install trustme

# Instrument
cd ../../../

FILES_TO_INSTRUMENT=$(find ./data/repos/$REPO_NAME/$2 -type f -name "*.py")

python3 -m lexecutor.Instrument --files $FILES_TO_INSTRUMENT --iids iids.json

# Discard tests that cannot be executed
# Request
# rm ./data/repos/$REPO_NAME/$3/conftest.py
# Run tests
cd ./data/repos/$REPO_NAME
pytest ./$3