# LExecutor

Paper (pre-print): https://arxiv.org/abs/2302.02343

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

## Fine-tune CodeT5 Model

1. Prepare the dataset, see `python -m lexecutor.predictors.codet5.PrepareData --help`

2. Run the training, see `python -m lexecutor.predictors.codet5.FineTune --help`

## Use Predictor

1. Set the LExecutor mode to PREDICT at Runtime.py
2. Instrument the Python files that will be LExecuted
3. Run the Python files instrumented in step 2

## Running the Pynguin Baseline

Extract functions into individual files, see:

`python -m lexecutor.evaluation.FunctionExtractor --help`

Create and enter a virtual environment for Python 3.10 (required by the newest Pynguin version):

`python3.10 -m venv myenv_py3.10`

`source myenv_3.10/bin/activate`

Run Pynguin on the extracted functions:

`python -m lexecutor.evaluation.RunPyngiun --files dir_with_functions/*.py`
