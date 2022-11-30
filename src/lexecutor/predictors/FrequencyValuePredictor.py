from .ValuePredictor import ValuePredictor
from .NaiveValuePredictor import NaiveValuePredictor
from ..TraceEntries import read_traces, NameEntry, CallEntry, AttributeEntry, BinOpEntry
from collections import Counter
from random import choices


class FrequencyValuePredictor(ValuePredictor):
    def __init__(self, trace_files):
        self.name_to_values = {}
        self.call_to_values = {}
        self.attribute_to_values = {}
        self.binop_to_values = {}

        self.naive_predictor = NaiveValuePredictor()  # as a fallback

        self.total_predictions = 0
        self.frequency_based_predictions = 0

        entries = read_traces(trace_files)
        for entry in entries:
            if isinstance(entry, NameEntry):
                key = entry.name
                self.name_to_values.setdefault(key, Counter())[
                    entry.value] += 1
            elif isinstance(entry, CallEntry):
                key = f"{entry.fct_name}--{entry.args}"
                self.call_to_values.setdefault(key, Counter())[
                    entry.value] += 1
            elif isinstance(entry, AttributeEntry):
                key = f"{entry.base}--{entry.attr_name}"
                self.attribute_to_values.setdefault(key, Counter())[
                    entry.value] += 1
            elif isinstance(entry, BinOpEntry):
                key = f"{entry.left}--{entry.operator}--{entry.right}"
                self.binop_to_values.setdefault(key, Counter())[
                    entry.value] += 1

    def name(self, iid, name):
        counter = self.name_to_values.get(name)
        self.total_predictions += 1
        if counter is None:
            return self.naive_predictor.name(iid, name)
        else:
            self.frequency_based_predictions += 1
            return choices(list(counter.keys()), list(counter.values()))[0]

    def call(self, iid, fct, *args, **kwargs):
        key = f"{fct}--{args}"
        counter = self.call_to_values.get(key)
        self.total_predictions += 1
        if counter is None:
            return self.naive_predictor.call(iid, fct, *args, **kwargs)
        else:
            self.frequency_based_predictions += 1
            return choices(list(counter.keys()), list(counter.values()))[0]

    def attribute(self, iid, base, attr_name):
        key = f"{base}--{attr_name}"
        counter = self.attribute_to_values.get(key)
        self.total_predictions += 1
        if counter is None:
            return self.naive_predictor.attribute(iid, base, attr_name)
        else:
            self.frequency_based_predictions += 1
            return choices(list(counter.keys()), list(counter.values()))[0]

    def binary_operation(self, iid, left, operator, right):
        key = f"{left}--{operator}--{right}"
        counter = self.binop_to_values.get(key)
        self.total_predictions += 1
        if counter is None:
            return self.naive_predictor.binary_operation(iid, left, operator, right)
        else:
            self.frequency_based_predictions += 1
            return choices(list(counter.keys()), list(counter.values()))[0]

    def print_stats(self):
        print(f"{self.frequency_based_predictions}/{self.total_predictions} ({self.frequency_based_predictions/self.total_predictions if self.total_predictions > 0 else 0}) predictions were frequency based")
