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

## Usage Guide

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

#### Value-use events

To gather a corpus of value-use events for training the neural model, we proceed as follows:

1. Set the LExecutor mode to RECORD at `Runtime.py`
2. Execute `chmod +x get_traces.sh`
3. Execute `get_traces.sh` giving the required arguments, e.g. `get_traces https://github.com/Textualize/rich rich tests`

The output is stored as follows: the repositories with instrumented files and trace files is stored in `./data/repos` and the instruction ids is stored in `./iids.json`.

#### Open-source functions

To gather a dataset of functions extracted from open-source Python projects, we proceed as follows:

1. Execute `chmod +x get_function_bodies_dataset.sh`
2. Execute `get_function_bodies_dataset.sh`

The output, i.e. the repositories and respective randomly selected functions, is stored in `./data/repos` and `./popular_projects_snippets_dataset`, respectively.

#### Stack Overflow snippets

To gather a dataset of code snippets from Stack Overflow, we execute `python get_stackoverflow_snippets_dataset.py --dest_dir so_snippets_dataset`

The output, i.e. the code snippets from Stack Overflow, is stored in `./so_snippets_dataset`.

### Model training

Our current implementation integrates two pre-trained models, CodeT5 and CodeBERT, which we fine-tune for our prediction task as follows.

#### CodeT5

1. Prepare the dataset running `python -m lexecutor.predictors.codet5.PrepareData --iids iids.json --traces traces.txt --output_suffix _codeT5`

The output is stored in `./train_codeT5.pt` and `./validate_codeT5.pt`.

2. Fine-tune the model executing `python -m lexecutor.predictors.codet5.FineTune --train_tensors train_codeT5.pt --validate_tensors validate_codeT5.pt --output_dir ./data/codeT5_models`

The output is stored in `./data/codeT5_models`.

#### CodeBERT

1. Prepare the dataset running `python -m lexecutor.predictors.codebert.PrepareData --iids iids.json --traces traces.txt --output_suffix _codeBERT`

The output is stored in `./train_codeBERT.pt` and `./validate_codeBERT.pt`.

2. Fine-tune the model executing `python -m lexecutor.predictors.codeBERT.FineTune --train_tensors train_codeBERT.pt --validate_tensors validate_codeBERT.pt --output_dir ./data/codeBERT_models`

The output is stored in `./data/codeBERT_models`.

### Baselines

### Pynguin

Extract functions into individual files, see:

`python -m lexecutor.evaluation.FunctionExtractor --help`

Create and enter a virtual environment for Python 3.10 (required by the newest Pynguin version):

`python3.10 -m venv myenv_py3.10`

`source myenv_3.10/bin/activate`

Run Pynguin on the extracted functions:

`python -m lexecutor.evaluation.RunPyngiun --files dir_with_functions/*.py`


