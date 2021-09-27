from Trace import Trace
from ValueAbstraction import restore_value
import atexit

# ------- begin: select mode -----
# mode = "RECORD"    # record values and write into a trace file
# mode = "PREDICT"   # predict and inject values if missing in exeuction
mode = "REPLAY"  # replay a previously recorded trace (mostly for testing)
# ------- end: select mode -------

if mode == "RECORD":
    trace = Trace("trace.out")
    atexit.register(lambda: trace.flush())
elif mode == "REPLAY":
    with open("trace.out", "r") as file:
        trace = file.readlines()
    next_trace_idx = 0


print(f"### LExecutor running in {mode} mode ###")


def _n_(iid, name, lambada):
    return mode_branch(iid, lambada, lambda v: trace.append_name(iid, name, v))


def _c_(iid, fct, *args):
    return mode_branch(iid, fct, lambda v: trace.append_call(iid, fct, args, v), *args)


def _a_(iid, base, attr_name):
    return mode_branch(iid, getattr, lambda v: trace.append_attribute(iid, base, attr_name, v), base, attr_name)


def _b_(iid, left, operator, right):
    return mode_branch(iid, perform_binary_op, lambda v: trace.append_binary_operator(iid, left, operator, right, v), left, operator, right)


def perform_binary_op(left, operator, right):
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
    return v


def mode_branch(iid, perform_fct, record_fct, *perform_fct_args):
    if mode in ("RECORD", "PREDICT"):
        try:
            v = perform_fct(*perform_fct_args)
            if mode == "RECORD":
                record_fct(v)
            return v
        except Exception as e:
            if mode == "PREDICT":
                # TODO: inject based in trained model
                return 23
            else:
                raise e
    elif mode == "REPLAY":
        # replay mode
        global next_trace_idx
        trace_line = trace[next_trace_idx]
        next_trace_idx += 1
        segments = trace_line.split(" ")
        trace_iid = int(segments[0])
        abstract_value = segments[-1]
        if iid != trace_iid:
            raise Exception(
                f"trace_iid={trace_iid} doesn't match execution iid={iid}")
        v = restore_value(abstract_value)
        print(f"At {iid} returning replay value: {v}")  # TODO RAD
        return v
    else:
        raise Exception(f"Unexpected mode {mode}")
