# LExecutor: Learning-Guided Execution

This repository contains the implementation of LExecutor and supplementary material for the paper "LExecutor: Learning-Guided Execution" (FSE'23).

Paper (pre-print): https://arxiv.org/abs/2302.02343

## Installation Guide

Create and enter a virtual environment:

`virtualenv -p /usr/bin/python3.8 myenv`

`source myenv/bin/activate`

Install requirements:

`pip install -r requirements.txt`

Locally install the package in development/editable mode:

`pip install -e ./`

## LExecutor Usage Guide

1. Instrument the Python files that will be LExecuted
2. Run the Python files instrumented in step 1

As a simple example, consider that the following code is in ./files/file.py. 

```python
if (not has_min_size(all_data)):
    raise RuntimeError("not enough data")

train_len = round(0.8 * len(all_data))

logger.info(f"Extracting training data with {config_str}")

train_data = all_data[0:train_len]
```
Then, to *LExecute* the code, do as follows:

1. Instrument the code:
`python -m lexecutor.Instrument --files ./files/file.py`

2. Run the instrumented code:
`python ./files/file.py`

## Replication Guide

### Datasets

### Value-use events

### Open-source functions

### Stack Overflow snippets

### Baselines

### Pynguin

Extract functions into individual files, see:

`python -m lexecutor.evaluation.FunctionExtractor --help`

Create and enter a virtual environment for Python 3.10 (required by the newest Pynguin version):

`python3.10 -m venv myenv_py3.10`

`source myenv_3.10/bin/activate`

Run Pynguin on the extracted functions:

`python -m lexecutor.evaluation.RunPyngiun --files dir_with_functions/*.py`

## Fine-tuning Guide

### CodeT5 Model

1. Prepare the dataset, see `python -m lexecutor.predictors.codet5.PrepareData --help`

2. Run the training, see `python -m lexecutor.predictors.codet5.FineTune --help`
