from Trace import Trace
from ValueAbstraction import restore_value
import atexit

# ------- begin: select mode -----
mode = "RECORD"    # record values and write into a trace file
trace = Trace("trace.out")
atexit.register(lambda: trace.flush())

# mode = "PREDICT"   # predict and inject values if missing in exeuction

# mode = "REPLAY"  # replay a previously recorded trace (mostly for testing)
# with open("trace.out", "r") as file:
#     trace = file.readlines()
# next_trace_idx = 0
# ------- end: select mode -------

print(f"### LExecutor running in {mode} mode ###")


def _n_(iid, name, lambada):
    v = lambada()
    trace.append_name(iid, name, v)
    return v


def _c_(iid, fct, *args):
    v = fct(*args)
    trace.append_call(iid, fct, args, v)
    return v


def _a_(iid, base, attr_name):
    v = getattr(base, attr_name)
    trace.append_attribute(iid, base, attr_name, v)
    return v


def _b_(iid, left, operator, right):
    # boolean operators
    if operator == "And":
        v = left and right
    elif operator == "Or":
        v = left or right
    # arithmetic operators
    elif operator == "Add":
        v = left + right
    elif operator == "BitAnd":
        v = left & right
    elif operator == "BitOr":
        v = left | right
    elif operator == "BitXor":
        v = left ^ right
    elif operator == "Divide":
        v = left / right
    elif operator == "FloorDivide":
        v = left // right
    elif operator == "LeftShift":
        v = left << right
    elif operator == "MatrixMultiply":
        v = left @ right
    elif operator == "Modulo":
        v = left % right
    elif operator == "Multiply":
        v = left * right
    elif operator == "Power":
        v = left ^ right
    elif operator == "RightShift":
        v = left >> right
    elif operator == "Subtract":
        v = left - right
    # comparison operators
    elif operator == "Equal":
        v = left == right
    elif operator == "GreaterThan":
        v = left > right
    elif operator == "GreaterThanEqual":
        v = left >= right
    elif operator == "In":
        v = left in right
    elif operator == "Is":
        v = left is right
    elif operator == "LessThan":
        v = left < right
    elif operator == "LessThanEqual":
        v = left <= right
    elif operator == "NotEqual":
        v = left != right
    elif operator == "IsNot":
        v = left is not right
    elif operator == "NotIn":
        v = left not in right
    else:
        raise Exception(f"Unexpected binary operator: {operator}")
    trace.append_binary_operator(iid, left, operator, right, v)


def _lexecutor_(value, iid):
    if mode in ("RECORD", "PREDICT"):
        try:
            v = value()
            print(f"{iid}: Got actual value {v}")
            if mode == "RECORD":
                trace.append(v, iid)
            return v
        except Exception as e:
            if mode == "PREDICT":
                # TODO: inject based in trained model
                v = 1
                print(f"{iid}: Will return default value {v} because {e}")
                return v
            else:
                raise e
    elif mode == "REPLAY":
        # replay mode
        global next_trace_idx
        trace_line = trace[next_trace_idx]
        next_trace_idx += 1
        trace_iid, abstract_value = trace_line.split(" ", 1)
        trace_iid = int(trace_iid)
        if iid != trace_iid:
            raise Exception(
                f"trace_iid={trace_iid} doesn't match execution iid={iid}")
        v = restore_value(abstract_value)
        return v

    else:
        raise Exception(f"Unexpected mode {mode}")
