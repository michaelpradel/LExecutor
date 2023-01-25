from .Logging import logger
from .Hyperparams import Hyperparams as params
import random


def abstract_value(value):
    t = type(value)
    # common primitive values
    if value is None:
        abtract_value = "@None"
    elif value is True:
        abtract_value = "@True"
    elif value is False:
        abtract_value = "@False"
    # strings
    elif t is str:
        if len(value) == 0:
            abtract_value = "@str_empty"
        else:
            abtract_value = "@str_nonempty"
    # built-in numeric types
    elif t is int:
        if value < 0:
            abtract_value = "@int_neg"
        elif value == 0:
            abtract_value = "@int_zero"
        else:
            abtract_value = "@int_pos"
    elif t is float:
        if value < 0:
            abtract_value = "@float_neg"
        elif value == 0:
            abtract_value = "@float_zero"
        else:
            abtract_value = "@float_pos"
    # built-in sequence types
    elif t is list:
        if len(value) == 0:
            abtract_value = "@list_empty"
        else:
            abtract_value = "@list_nonempty"
    elif t is tuple:
        if len(value) == 0:
            abtract_value = "@tuple_empty"
        else:
            abtract_value = "@tuple_nonempty"
    # built-in set and dict types
    elif t is set:
        if len(value) == 0:
            abtract_value = "@set_empty"
        else:
            abtract_value = "@set_nonempty"
    elif t is dict:
        if len(value) == 0:
            abtract_value = "@dict_empty"
        else:
            abtract_value = "@dict_nonempty"
    # functions and methods
    elif callable(value):
        if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
            abtract_value = "@resource"
        else:
            abtract_value = "@callable"
    # all other types
    else:
        abtract_value = "@object"

    return abtract_value, str(t)[:20]


class DummyResource(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        return True


class DummyObject():
    def __init__(self, *a, **b):
        pass


fine_to_coarse_grained = {
    "@None": "@None",
    "@True": "@bool",
    "@False": "@bool",
    "@str_empty": "@str",
    "@str_nonempty": "@str",
    "@int_neg": "@int",
    "@int_zero": "@int",
    "@int_pos": "@int",
    "@float_neg": "@float",
    "@float_zero": "@float",
    "@float_pos": "@float",
    "@list_empty": "@list",
    "@list_nonempty": "@list",
    "@tuple_empty": "@tuple",
    "@tuple_nonempty": "@tuple",
    "@set_empty": "@set",
    "@set_nonempty": "@set",
    "@dict_empty": "@dict",
    "@dict_nonempty": "@dict",
    "@resource": "@resource",
    "@callable": "@callable",
    "@object": "@object",
}


if params.value_abstraction.startswith("coarse-grained"):
    if params.value_abstraction == "coarse-grained-deterministic":
        def restore_value(abstract_value):
            abstract_value = fine_to_coarse_grained["@" + abstract_value][1:]
            # common primitive values
            if abstract_value == "None":
                return None
            elif abstract_value == "bool":
                return True
            # strings
            elif abstract_value == "str":
                return "a"
            # built-in numeric types
            elif abstract_value == "int":
                return 1
            elif abstract_value == "float":
                return 1.0
            # built-in sequence types
            elif abstract_value == "list":
                return [DummyObject()]
            elif abstract_value == "tuple":
                return (DummyObject(),)
            # built-in set and dict types
            elif abstract_value == "set":
                return {DummyObject()}
            elif abstract_value == "dict":
                return {"a": DummyObject()}
            # functions and methods
            elif abstract_value == "resource":
                return DummyResource()
            elif abstract_value == "callable":
                return DummyObject
            elif abstract_value == "object":
                return DummyObject()
            # all other types
            else:
                logger.info("Unknown abstract value: %s", abstract_value)
                return DummyObject()
    elif params.value_abstraction == "coarse-grained-randomized":
        def restore_value(abstract_value):
            abstract_value = fine_to_coarse_grained["@" + abstract_value][1:]
            # common primitive values
            if abstract_value == "None":
                return None
            elif abstract_value == "bool":
                return random.choice([True, False])
            # strings
            elif abstract_value == "str":
                return random.choice(["", "a"])
            # built-in numeric types
            elif abstract_value == "int":
                return random.choice([-1, 0, 1])
            elif abstract_value == "float":
                return random.choice([-1.0, 0.0, 1.0])
            # built-in sequence types
            elif abstract_value == "list":
                return random.choice([[], [DummyObject()]])
            elif abstract_value == "tuple":
                return random.choice([(), (DummyObject(),)])
            # built-in set and dict types
            elif abstract_value == "set":
                return random.choice([{}, {DummyObject()}])
            elif abstract_value == "dict":
                return random.choice([{}, {"a": DummyObject()}])
            # functions and methods
            elif abstract_value == "resource":
                return DummyResource()
            elif abstract_value == "callable":
                return DummyObject
            elif abstract_value == "object":
                return DummyObject()
            # all other types
            else:
                logger.info("Unknown abstract value: %s", abstract_value)
                return DummyObject()

elif params.value_abstraction == "fine-grained":
    def restore_value(abstract_value):
        # common primitive values
        if abstract_value == "None":
            return None
        elif abstract_value == "True":
            return True
        elif abstract_value == "False":
            return False
        # strings
        elif abstract_value == "str_empty":
            return ""
        elif abstract_value == "str_nonempty":
            return "a"
        # built-in numeric types
        elif abstract_value == "int_neg":
            return -1
        elif abstract_value == "int_zero":
            return 0
        elif abstract_value == "int_pos":
            return 1
        elif abstract_value == "float_neg":
            return -1.0
        elif abstract_value == "float_zero":
            return 0.0
        elif abstract_value == "float_pos":
            return 1.0
        # built-in sequence types
        elif abstract_value == "list_empty":
            return []
        elif abstract_value == "list_nonempty":
            return [DummyObject()]
        elif abstract_value == "tuple_empty":
            return ()
        elif abstract_value == "tuple_nonempty":
            return (DummyObject(),)
        # built-in set and dict types
        elif abstract_value == "set_empty":
            return set()
        elif abstract_value == "set_nonempty":
            return {DummyObject()}
        elif abstract_value == "dict_empty":
            return {}
        elif abstract_value == "dict_nonempty":
            return {"a": DummyObject()}
        # functions and methods
        elif abstract_value == "resource":
            return DummyResource()
        elif abstract_value == "callable":
            return DummyObject
        elif abstract_value == "object":
            return DummyObject()
        # all other types
        else:
            logger.info("Unknown abstract value: %s", abstract_value)
            return DummyObject()

else:
    raise ValueError(
        f"Unknown setting for value_abstraction: {params.value_abstraction}")
