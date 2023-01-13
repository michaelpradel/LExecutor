# LExecutor: DO NOT INSTRUMENT

from lexecutor.Runtime import _n_
from lexecutor.Runtime import _a_
from lexecutor.Runtime import _c_
if (not _c_(770559, _n_(770557, "has_min_size", lambda: has_min_size), _n_(770558, "all_data", lambda: all_data))):
    raise _c_(770561, _n_(770560, "RuntimeError", lambda: RuntimeError), "not enough data")

train_len = _c_(770566, _n_(770562, "round", lambda: round), 0.8 * _c_(770565, _n_(770563, "len", lambda: len), _n_(770564, "all_data", lambda: all_data)))

_c_(770570, _a_(770568, _n_(770567, "logger", lambda: logger), "info"), f"Extracting training data with config {_n_(770569, 'config_str', lambda: config_str)}")

train_data = _n_(770571, "all_data", lambda: all_data)[0:_n_(770572, "train_len", lambda: train_len)]
_c_(770575, _n_(770573, "print", lambda: print), _n_(770574, "train_data", lambda: train_data))