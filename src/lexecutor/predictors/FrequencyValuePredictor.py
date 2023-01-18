from .ValuePredictor import ValuePredictor
from .NaiveValuePredictor import NaiveValuePredictor
from .codet5.PrepareData import read_traces, clean_entries
from collections import Counter
from random import choices


class FrequencyValuePredictor(ValuePredictor):
    def __init__(self, trace_files):
        self.name_to_values = {}
        self.call_to_values = {}
        self.attribute_to_values = {}

        self.naive_predictor = NaiveValuePredictor()  # as a fallback

        self.total_predictions = 0
        self.frequency_based_predictions = 0

        entries = read_traces(trace_files)
        clean_entries(entries)
        for index, entry in entries.iterrows():
            key = entry["name"]
            if entry["kind"] == "name":
                self.name_to_values.setdefault(key, Counter())[
                    entry.value] += 1
            elif entry["kind"] == "call":
                self.call_to_values.setdefault(key, Counter())[
                    entry.value] += 1
            elif entry["kind"] == "attribute":
                self.attribute_to_values.setdefault(key, Counter())[
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

    def print_stats(self):
        print(f"{self.frequency_based_predictions}/{self.total_predictions} ({self.frequency_based_predictions/self.total_predictions if self.total_predictions > 0 else 0}) predictions were frequency based")
