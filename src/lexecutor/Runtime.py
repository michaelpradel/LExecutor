import atexit
import sys
from .Hyperparams import Hyperparams as params
from .TraceWriter import TraceWriter
from .ValueAbstraction import restore_value, DummyObject
# from .predictors.AsIs import AsIs
# from .predictors.NaiveValuePredictor import NaiveValuePredictor
# from .predictors.FrequencyValuePredictor import FrequencyValuePredictor
# from .predictors.feedforward.NeuralValuePredictor import NeuralValuePredictor
from .predictors.codet5.CodeT5ValuePredictor import CodeT5ValuePredictor
from .RuntimeStats import RuntimeStats
from .Logging import logger


logger.info("Runtime starting")

# ------- begin: select mode -----
# mode = "RECORD"    # record values and write into a trace file
mode = "PREDICT"   # predict and inject values if missing in exeuction
# mode = "REPLAY"  # replay a previously recorded trace (mostly for testing)
# ------- end: select mode -------


if mode == "RECORD":
    trace = TraceWriter()
    atexit.register(lambda: trace.write_to_file())
    runtime_stats = None
elif mode == "PREDICT":
    # predictor = AsIs()
    # predictor = NaiveValuePredictor()
    # predictor = FrequencyValuePredictor("/home/beatriz/LExecutor/all_training_traces.txt")
    # predictor = NeuralValuePredictor()

    runtime_stats = RuntimeStats()
    atexit.register(runtime_stats.print)
    predictor = CodeT5ValuePredictor(runtime_stats)

    # # for running experiments
    file = sys.argv[0]
    atexit.register(runtime_stats.save, file, predictor.__class__.__name__)
elif mode == "REPLAY":
    with open("trace.out", "r") as file:
        trace = file.readlines()
    next_trace_idx = 0
    runtime_stats = None

logger.info(f"### LExecutor running in {mode} mode ###")


def _n_(iid, name, lambada):
    if params.verbose:
        logger.info(f"\nAt iid={iid}, looking up name '{name}'")

    if runtime_stats is not None:
        runtime_stats.total_uses += 1
        runtime_stats.cover_iid(iid)

    perform_fct = lambada

    def record_fct(v):
        trace.append_name(iid, name, v)

    def predict_fct():
        return predictor.name(iid, name)

    return mode_branch(iid, perform_fct, record_fct, predict_fct, kind="name")


def _c_(iid, fct, *args, **kwargs):
    if params.verbose:
        logger.info(f"\nAt iid={iid}, calling function {fct}")

    if runtime_stats is not None:
        runtime_stats.total_uses += 1
        runtime_stats.cover_iid(iid)

    def perform_fct():
        return fct(*args, **kwargs)

    def record_fct(v):
        trace.append_call(iid, fct, args, kwargs, v)

    def predict_fct():
        return predictor.call(iid, fct, args, kwargs)

    kind = "call_dummy" if fct is DummyObject else "call"
    return mode_branch(iid, perform_fct, record_fct, predict_fct, kind=kind)


def _a_(iid, base, attr_name):
    if params.verbose:
        logger.info(f"\nAt iid={iid}, looking up attribute '{attr_name}'")

    if runtime_stats is not None:
        runtime_stats.total_uses += 1
        runtime_stats.cover_iid(iid)

    def perform_fct():
        # return getattr(base, attr_name)
        # unmangle private attributes (code copied from DynaPyt)
        if (attr_name.startswith('__')) and (not attr_name.endswith('__')):
            if type(base).__name__ == 'type':
                parents = [base]
            else:
                parents = [type(base)]
            found = True
            while len(parents) > 0:
                found = True
                cur_par = parents.pop()
                try:
                    cur_name = cur_par.__name__
                    cur_name = cur_name.lstrip('_')
                    return getattr(base, '_'+cur_name+attr_name)
                except AttributeError:
                    found = False
                    parents.extend(list(cur_par.__bases__))
                    continue
                break
            if not found:
                raise AttributeError()
        else:
            return getattr(base, attr_name)

    def record_fct(v):
        trace.append_attribute(iid, base, attr_name, v)

    def predict_fct():
        return predictor.attribute(iid, base, attr_name)

    return mode_branch(iid, perform_fct, record_fct, predict_fct, kind="attribute")


def mode_branch(iid, perform_fct, record_fct, predict_fct, kind):
    if mode == "RECORD":
        v = perform_fct()
        record_fct(v)
        return v
    elif mode == "PREDICT":
        if kind == "call_dummy":
            # predict and inject a return value
            v = predict_fct()
            return v
        else:
            # try to perform the regular behavior and intervene in case of exceptions caused by missing values
            try:
                v = perform_fct()
                if params.verbose:
                    logger.info("Found/computed/returned regular value")
                return v
            except Exception as e:
                if (type(e) == NameError and kind == "name") \
                        or (type(e) == AttributeError and kind == "attribute"):
                    if params.verbose:
                        logger.info(
                            f"Catching '{type(e)}' during {kind} and calling predictor instead")
                    v = predict_fct()
                    runtime_stats.guided_uses += 1
                    return v
                else:
                    if params.verbose:
                        logger.info(
                            f"Exception '{type(e)}' not caught, re-raising")
                    runtime_stats.uncaught_exception(iid, e)
                    raise e
    elif mode == "REPLAY":
        # replay mode
        global next_trace_idx
        trace_line = trace[next_trace_idx].rstrip()
        next_trace_idx += 1
        segments = trace_line.split(" ")
        trace_iid = int(segments[0])
        abstract_value = segments[-1]
        if iid != trace_iid:
            raise Exception(
                f"trace_iid={trace_iid} doesn't match execution iid={iid}")
        v = restore_value(abstract_value)
        return v
    else:
        raise Exception(f"Unexpected mode {mode}")


def print_prediction_stats():
    pass
