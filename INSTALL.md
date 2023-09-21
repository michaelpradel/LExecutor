# Installation Guide

Create and enter a virtual environment:

```
virtualenv -p /usr/bin/python3.8 myenv
source myenv/bin/activate
```

Install requirements:

```
pip install -r requirements.txt
```

Locally install the package in development/editable mode:

```
pip install -e ./
```

# Usage Guide

1. Instrument the Python files that will be LExecuted

2. Run the Python files instrumented in step 1

As a simple example, consider that the following code is in `./files/file.py`. 

```python
if (not has_min_size(all_data)):
    raise RuntimeError("not enough data")

train_len = round(0.8 * len(all_data))

logger.info(f"Extracting training data with {config_str}")

train_data = all_data[0:train_len]
```
Then, to *LExecute* the code, do as follows:

1. Instrument the code:
```
python -m lexecutor.Instrument --files ./files/file.py
```

2. Run the instrumented code:
```
python ./files/file.py
```
