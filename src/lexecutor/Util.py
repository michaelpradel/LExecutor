import torch as t
from datetime import datetime

dtype = t.cuda.float if t.cuda.is_available() else t.float
device = "cuda" if t.cuda.is_available() else "cpu"


def gather_files(files_arg):
    if len(files_arg) == 1 and files_arg[0].endswith(".txt"):
        files = []
        with open(files_arg[0]) as fp:
            for line in fp.readlines():
                files.append(line.rstrip())
    else:
        for f in files_arg:
            if not f.endswith(".py"):
                raise Exception(f"Incorrect argument, expected .py file: {f}")
        files = files_arg
    return files


def timestamp():
    epoch = datetime.utcfromtimestamp(0)
    now = datetime.now()
    return round((now-epoch).total_seconds()*1000000.0)
