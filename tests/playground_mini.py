# LExecutor: DO NOT INSTRUMENT

from lexecutor.runtime import _n_
from lexecutor.runtime import _a_
from lexecutor.runtime import _c_
from lexecutor.runtime import _b_
class C:
    a = [1, 2]
    b = _c_(4, _n_(2, "set", lambda set = set: set), _n_(3, "a", lambda a = a: a))
    c = None

c = _c_(6, _n_(5, "C", lambda C = C: C))
_c_(9, _n_(7, "print", lambda print = print: print), _n_(8, "c", lambda c = c: c))