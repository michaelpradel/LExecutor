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
            raise Exception(f"trace_iid={trace_iid} doesn't match execution iid={iid}")
        v = restore_value(abstract_value)
        return v

    else:
        raise Exception(f"Unexpected mode {mode}")
