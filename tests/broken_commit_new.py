# LExecutor: DO NOT INSTRUMENT

from lexecutor.Runtime import _n_
from lexecutor.Runtime import _a_
from lexecutor.Runtime import _c_
if (not _c_(770542, _n_(770540, "has_min_size", lambda: has_min_size), _n_(770541, "all_data", lambda: all_data))):
    raise _c_(770544, _n_(770543, "RuntimeError", lambda: RuntimeError), "not enough data")

train_len = 0.8 * _c_(770547, _n_(770545, "len", lambda: len), _n_(770546, "all_data", lambda: all_data))

_c_(770551, _a_(770549, _n_(770548, "logger", lambda: logger), "info"), f"Extracting training data with config {_n_(770550, 'config_str', lambda: config_str)}")

train_data = _n_(770552, "all_data", lambda: all_data)[0:_n_(770553, "train_len", lambda: train_len)]
_c_(770556, _n_(770554, "print", lambda: print), _n_(770555, "train_data", lambda: train_data))