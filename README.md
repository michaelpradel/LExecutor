# LExecutor: Learning-Guided Execution

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8263493.svg)](https://doi.org/10.5281/zenodo.8263493)

This repository contains the implementation of LExecutor and supplementary material for the paper "LExecutor: Learning-Guided Execution" (FSE'23).

Paper (pre-print): https://arxiv.org/abs/2302.02343

## Getting Started Guide

1. Check that your setup meets the [REQUIREMENTS.md](REQUIREMENTS.md).
2. Follow the installation instructions in [INSTALL.md](INSTALL.md).

## Replication Guide

To reproduce results from the paper, follow these instructions:

First, install LExecutor using the instructions above.

### Accuracy of the Neural Model (RQ1)

#### Value-use events dataset

To gather a corpus of value-use events for training and evaluating the neural model, we proceed as follows:

1. Set the LExecutor mode to RECORD at `./src/lexecutor/Runtime.py`

2. Make `get_traces.sh` executable:
```
chmod +x get_traces.sh
```

3. For every considered project, execute `get_traces.sh` giving the required arguments, e.g.:
```
./get_traces.sh https://github.com/Textualize/rich rich tests
```

4. Get the path of all the generated traces:
```
find ./data/repos/ -type f -name "trace_*.h5" > traces.txt
```

The output is stored as follows: the repositories with instrumented files and trace files are stored in `./data/repos`; the instruction ids is stored in `./iids.json`; the trace paths are stored in `./traces.txt`.

#### Model training and validation

Our current implementation integrates two pre-trained models, CodeT5 and CodeBERT, which we fine-tune for our prediction task as follows.

##### CodeT5

1. Create a folder to store the output:
```
mkdir ./data/codeT5_models_fine-grained
```

2. Prepare the dataset:
```
python -m lexecutor.predictors.codet5.PrepareData \
  --iids iids.json \
  --traces traces.txt \
  --output_dir ./data/codeT5_models_fine-grained
```

3. Fine-tune the model:
```
python -m lexecutor.predictors.codet5.FineTune \
  --train_tensors ./data/codeT5_models_fine-grained/train.pt \
  --validate_tensors ./data/codeT5_models_fine-grained/validate.pt \
  --output_dir ./data/codeT5_models_fine-grained \
  --stats_dir ./data/codeT5_models_fine-grained
```

The output, i.e. the tensors, models for every epoch, training loss and validation accuracy, is stored in `./data/codeT5_models_fine-grained`.

##### CodeBERT

1. Create a folder to store the output:
```
mkdir ./data/codeBERT_models_fine-grained
```

2. Prepare the dataset:
```
python -m lexecutor.predictors.codebert.PrepareData \
  --iids iids.json \
  --traces traces.txt \
  --output_dir ./data/codeBERT_models_fine-grained
```

3. Fine-tune the model:
```
python -m lexecutor.predictors.codeBERT.FineTune \
  --train_tensors ./data/codeBERT_models_fine-grained/train.pt \
  --validate_tensors ./data/codeBERT_models_fine-grained/validate.pt \
  --output_dir ./data/codeBERT_models_fine-grained \
  --stats_dir ./data/codeBERT_models_fine-grained
```

The output, i.e. the tensors, the models for every epoch, training loss and validation accuracy, is stored in `./data/codeBERT_models_fine-grained`.

By default, we train and use the models based on the fine-grained abstraction of values. To fine-tune the models based on the coarse-grained abstraction of values, set `value_abstraction` to `coarse-grained-deterministic` or `coarse-grained-randomized` in `./src/LExecutor/Hyperparams.py`. Then, replace `fine-grained` by `coarse-grained` in the steps 1-3 above. 

### Effectiveness at Covering Code and Efficiency at Guiding Executions (RQ2 and RQ3)

#### Datasets

##### Open-source functions

To gather a dataset of functions extracted from open-source Python projects, we proceed as follows:

1. Make `get_function_bodies_dataset.sh` executable:
```
chmod +x get_function_bodies_dataset.sh
```

2. Execute `get_function_bodies_dataset.sh`:
```
./get_function_bodies_dataset.sh
```

The output contains two extra versions of each function to fit the considered baseline approaches: 1) for functions that are methods, we wrapp them in a `Wrapper` class, otherwise we would not be able run Pynguin on them; 2) we add a function invocation to each function for them to be executed. This is required to run the code inside each function when running the baseline predictor based on Type4Py.

The output is stored as follows: the repositories are stored in `./data/repos`; the randomly selected functions are stored in `./popular_projects_snippets_dataset`; the paths to the files in each version of the dataset are stored in `popular_projects_function_bodies_dataset.txt`, `popular_projects_functions_dataset.txt` and `popular_projects_functions_with_invocation_dataset.txt`. Finally, auxiliary information useful to calculate line coverage afterwards are stored in `wrapp_info.csv` and `aux_data_functions_with_invocation_dataset.csv`.

##### Stack Overflow snippets

To gather a dataset of code snippets from Stack Overflow, we proceed as follows:

1. Create a folder to store the code snippets:
```
mkdir so_snippets_dataset
```

2. Get the code snippets:
```
python get_stackoverflow_snippets_dataset.py --dest_dir so_snippets_dataset
```

3. Get the path of all the collected snippets:
```
find ./so_snippets_dataset -type f -name "*.py" > so_snippets_dataset.txt
```

The output is stored as follows: the code snippets from Stack Overflow are stored in `./so_snippets_dataset` and their paths are stored in `so_snippets_dataset.txt`.

#### Data generation

1. Set the dataset under evaluation at `./src/LExecutor/Hyperparemeters.py`

2. Calculate the total lines in each file on the dataset under evaluation, e.g.:
```
python -m lexecutor.evaluation.CountTotalLines --files popular_projects_function_bodies_dataset.txt
```

3. Instrument the files in the dataset under evaluation, e.g.:
```
python -m lexecutor.Instrument --files popular_projects_function_bodies_dataset.txt --iids iids.json
```

4. Execute each predictor/baseline on the dataset under evaluation as follows:

   1. Set `./src/LExecutor/Runtime.py` to use the desired predictor. Some predictors/baselines require additional steps:
      - For the predictors based on CodeT5 and CodeBERT, the value abstraction must also be set at `./src/LExecutor/Hyperparemeters.py`
      - For the predictor based on Type4Py, make sure that the docker image containing Type4Py's pre-trained model is running according to [this tutorial](https://github.com/saltudelft/type4py/wiki/Type4Py's-Local-Model)
      - For the Pynguin baseline, execute the following steps:
           1. Create and enter a virtual environment for Python 3.10 (required by the newest Pynguin version):
               ```
               python3.10 -m venv myenv_py3.10
               source myenv_3.10/bin/activate
               ```

           2. Generate tests with Pynguin for the extracted functions:
               ```
               mkdir pynguin_tests
               python -m lexecutor.evaluation.RunPyngiun --files popular_projects_functions_dataset.txt --dest pynguin_tests
               ```

           3. Get the path of all the generated tests:
               ```
               find ./pynguin_tests -type f -name "test_*.py" > pynguin_tests.txt
               ```

           4. Set the predictor to `AsIs` and the file_type to `TESTE` in `./src/LExecutor/Runtime.py`
           
   2. Create a folder to store the log files, e.g.:
      ```
      mkdir logs
      mkdir logs/popular_projects_functions_dataset
      mkdir logs/popular_projects_functions_dataset/RandomPredictor
      ```

   3. Execute `RunExperiments.py` with the required arguments, e.g.:
      ```
      python -m lexecutor.evaluation.RunExperiments \
        --files popular_projects_functions_dataset.txt \
        --log_dest_dir logs/popular_projects_functions_dataset/RandomPredictor
      ```

      For the Pynguin baseline, make sure to include `--tests` and give the path to the generated tests, i.e. `pynguin_tests.txt`, to `--files` when executing `RunExperiments.py`

5. Process and combine the raw data generated:
   ```
   python -m lexecutor.evaluation.CombineData
   ```

#### Data analysis and plots generation

The code to get the plots for RQ2 and table content for RQ3 is available at `./src/notebooks/analyze_code_coverage_effectiveness_and_efficiency.ipynb`
  
### Using LExecutor to Find Semantics-Changing Commits (RQ4)

#### Pairs of old + new function from commits dataset

To gather a corpus of pairs of old + new function from commits, we proceed as follows:

1. Create a folder to store the function pairs for every considered project, e.g.:
```
mkdir data/function_pairs && mkdir data/function_pairs/flask
```

2. For every considered project, execute `FunctionPairExtractor.py` providing the required arguments, e.g.:
```
python -m lexecutor.evaluation.FunctionPairExtractor \
  --repo data/repos_with_commit_history/flask/ \
  --dest data/function_pairs/flask/
```

The output, i.e. the function pairs with code that invokes both functions and compares their return values, is stored in `compare.py` files under `data/function_pairs/`
#### Finding semantics-changing commits

1. Instrument the code in the `compare.py` files, e.g.:
```
python -m lexecutor.Instrument --files `find data/function_pairs/flask -name compare.py | xargs`
```

2. Run the instrumented code to compare its runtime behavior, e.g.:
```
for f in `find data/function_pairs/flask -name compare.py | xargs`; do timeout 30 python $f; done > out_flask
```



