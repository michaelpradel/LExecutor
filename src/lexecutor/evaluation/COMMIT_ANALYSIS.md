# Steps for applying LExecutor to find semantics-changing commits

Replace "flask" with the name of the repository you want to analyze.

## Extract pairs of old + new function from commits

```python -m lexecutor.evaluation.FunctionPairExtractor --repo data/repos/flask/ --dest data/function_pairs/flask/```

## Instrument the code

```python -m lexecutor.Instrument --files `find data/function_pairs/flask -name compare.py | xargs` ```

## Run the instrumented code to compare its runtime behavior

```for f in `find data/function_pairs/flask -name compare.py | xargs`; do timeout 30 python $f; done > out_flask```

