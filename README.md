# LExecutor

Create and enter a virtual environment:

`virtualenv -p /usr/bin/python3.8 myenv`

`source myenv/bin/activate`

Install requirements:

`pip install -r requirements.txt`

Locally install the package in development/editable mode:

`pip install -e ./`

Run the commands, e.g.,:

`python -m lexecutor.instrument --help`

Run the test suite of benchmark projects, e.g.,:

`cd data/repos/pandas`
`source myenv/bin/activate`
`pytest pandas/core/arrays/boolean.py`