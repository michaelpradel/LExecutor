import torch as t


dtype = t.float
device = "cuda" if t.cuda.is_available() else "cpu"
