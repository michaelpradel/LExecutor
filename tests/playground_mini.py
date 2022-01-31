# LExecutor: DO NOT INSTRUMENT

from lexecutor.runtime import _n_
from lexecutor.runtime import _a_
from lexecutor.runtime import _c_
from lexecutor.runtime import _b_
class C:
    def __init__(self):
        _c_(4, _a_(3, _n_(2, "super", lambda: super)(), "__init__"))

c = _c_(6, _n_(5, "C", lambda: C))
_c_(9, _n_(7, "print", lambda: print), _n_(8, "c", lambda: c))