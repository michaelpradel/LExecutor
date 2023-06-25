from .ValuePredictor import ValuePredictor
from ..ValueAbstraction import DummyObject, DummyResource
from ..Logging import logger
import random

class RandomPredictor(ValuePredictor):
    def __init__(self):
        super().__init__()
        self.values = [
            None,
            True,
            False,
            "",
            "a",
            -1,
            0,
            1,
            -1.0,
            0.0,
            1.0,
            [],
            [DummyObject()],
            (),
            (DummyObject(),),
            set(),
            {DummyObject()},
            {},
            {"a": DummyObject()},
            DummyResource(),
            DummyObject,
            DummyObject()
        ]
        
    def get_random_value(self):
        random_index = random.randint(0, len(self.values) - 1)
        return self.values[random_index]
    
    def name(self, iid, name):
        v = self.get_random_value()
        logger.info(f"{iid}: Predicting with RandomPredictor for name {name}: {v}")
        return v

    def call(self, iid, fct, fct_name, *args, **kwargs):
        v = self.get_random_value()
        logger.info(f"{iid}: Predicting with RandomPredictor for call: {v}")
        return v

    def attribute(self, iid, base, attr_name):
        v = self.get_random_value()
        logger.info(f"{iid}: Predicting with RandomPredictor for attribute {attr_name}: {v}")
        return v