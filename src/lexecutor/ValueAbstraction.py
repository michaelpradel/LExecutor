from .Logging import logger


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


class DummyCallable():
    def __init__(self, *a, **b):
        pass


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
        return [object()]
    elif abstract_value == "tuple_empty":
        return ()
    elif abstract_value == "tuple_nonempty":
        return (object(),)
    # built-in set and dict types
    elif abstract_value == "set_empty":
        return set()
    elif abstract_value == "set_nonempty":
        return {object()}
    elif abstract_value == "dict_empty":
        return {}
    elif abstract_value == "dict_nonempty":
        return {"a": object()}
    # functions and methods
    elif abstract_value == "resource":
        return DummyResource()
    elif abstract_value == "callable":
        return DummyCallable
    elif abstract_value == "object":
        return DummyCallable()
    # all other types
    else:
        logger.info("Unknown abstract value: %s", abstract_value)
        return DummyCallable()
