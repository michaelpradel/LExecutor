# LExecutor

## Setup

Create and enter a virtual environment:

`virtualenv -p /usr/bin/python3.8 myenv`

`source myenv/bin/activate`

Install requirements:

`pip install -r requirements.txt`

Locally install the package in development/editable mode:

`pip install -e ./`

Run the commands, e.g.,:

`python -m lexecutor.Instrument --help`

Run the test suite of benchmark projects, e.g.,:

`cd data/repos/pandas`
`source myenv/bin/activate`
`pytest pandas/core/arrays/boolean.py`

## Use Predictor

1. Set the LExecutor mode to PREDICT at Runtime.py
2. Instrument the Python files that will be LExecuted
3. Set the iids_file in CodeT5ValuePredictor.py to point to the iids file used or generated in step 2
4. Run the Python files instrumented in step 2
