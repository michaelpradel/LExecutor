from .ValuePredictor import ValuePredictor
from .NaiveValuePredictor import NaiveValuePredictor
from ..Logging import logger
from ..ValueAbstraction import restore_value
from random import choices
import json

class FrequencyValuePredictor(ValuePredictor):
    def __init__(self, values_frequencies_file):
        with open(f'{values_frequencies_file}', 'r') as openfile:
            values_frequencies = json.load(openfile)
    
        self.name_to_values = values_frequencies["name_to_values"]
        self.call_to_values = values_frequencies["call_to_values"]
        self.attribute_to_values = values_frequencies["attribute_to_values"]

        self.naive_predictor = NaiveValuePredictor()  # as a fallback

        self.total_predictions = 0
        self.frequency_based_predictions = 0

    def name(self, iid, name):
        counter = self.name_to_values.get(name)
        self.total_predictions += 1
        if counter is None:
            return self.naive_predictor.name(iid, name)
        else:
            self.frequency_based_predictions += 1
            v = choices(list(counter.keys()), list(counter.values()))[0]
            logger.info(f"{iid}: Predicting for name {name}: {v}")
            return restore_value(v)

    def call(self, iid, fct, fct_name, *args, **kwargs):
        counter = self.call_to_values.get(fct_name)
        self.total_predictions += 1
        if counter is None:
            return self.naive_predictor.call(iid, fct, *args, **kwargs)
        else:
            self.frequency_based_predictions += 1
            v = choices(list(counter.keys()), list(counter.values()))[0]
            logger.info(f"{iid}: Predicting for call: {v}")
            return restore_value(v)

    def attribute(self, iid, base, attr_name):
        counter = self.attribute_to_values.get(attr_name)
        self.total_predictions += 1
        if counter is None:
            return self.naive_predictor.attribute(iid, base, attr_name)
        else:
            self.frequency_based_predictions += 1
            v = choices(list(counter.keys()), list(counter.values()))[0]
            logger.info(f"{iid}: Predicting for attribute {attr_name}: {v}")
            return restore_value(v)

    def print_stats(self):
        print(f"{self.frequency_based_predictions}/{self.total_predictions} ({self.frequency_based_predictions/self.total_predictions if self.total_predictions > 0 else 0}) predictions were frequency based")
