from .ValuePredictor import ValuePredictor
from ..Logging import logger

class Toy:
    pass


class NaiveValuePredictor(ValuePredictor):
    def name(self, iid, name):
        v = Toy()
        logger.info(f"{iid}: Predicting for name {name}: {v}")
        return v

    def call(self, iid, fct, fct_name, *args, **kwargs):
        v = Toy()
        logger.info(f"{iid}: Predicting for call: {v}")
        return v

    def attribute(self, iid, base, attr_name):
        v = Toy()
        logger.info(f"{iid}: Predicting for attribute {attr_name}: {v}")
        return v

    def binary_operation(self, iid, left, operator, right):
        v = 3
        logger.info(f"{iid}: Predicting result of {operator} operation: {v}")
        return v
