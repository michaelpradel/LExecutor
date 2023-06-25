import torch as t
import numpy as np
from ..ValuePredictor import ValuePredictor
from ..DLUtil import device
from .CodeBERT import load_CodeBERT
from .InputFactory import InputFactory
from ...Logging import logger
import time
import requests
from requests.exceptions import ConnectionError
import subprocess
from ...ValueAbstraction import restore_value
from ...Hyperparams import Hyperparams as params
from ...IIDs import IIDs


class CodeBERTValuePredictor(ValuePredictor):
    def __init__(self, stats):
        self.stats = stats

        # load model
        self.tokenizer, self.model = load_CodeBERT()
        if params.value_abstraction == "fine-grained":
            model_path = "data/codeBERT_models/jan5_5_projects/pytorch_model.bin"
        elif params.value_abstraction == "coarse-grained-deterministic" or params.value_abstraction == "coarse-grained-randomized":
            model_path = "data/codeBERT_models/jan5_5_projects/pytorch_model.bin"
        self.model.load_state_dict(t.load(
            model_path, map_location=device))
        logger.info("CodeBERT model loaded")

        self.iids = IIDs(params.iids_file)
        self.stats = stats
        self.input_factory = InputFactory(self.iids, self.tokenizer)

    def _query_model(self, entry):
        # turn entry into vectors
        input_ids, _ = self.input_factory.entry_to_inputs(entry)
        input_ids = [tensor.cpu() for tensor in input_ids]

        # query the model and decode the result
        with t.no_grad():
            self.model.eval()
            generated_ids = self.model.generate(
                t.tensor(np.array([input_ids]), device=device))

        predicted_value = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True)

        if self.verbose:
            if self.tokenizer.bos_token_id not in generated_ids or self.tokenizer.eos_token_id not in generated_ids[0]:
                print(
                    f"Warning: CodeBERT likely produced a garbage value: {predicted_value}")

        return predicted_value, restore_value(predicted_value)
    



        val_as_string = response["v"]
        val = restore_value(val_as_string)

        return val_as_string, val

    def name(self, iid, name):
        entry = {"iid": iid, "name": name, "kind": "name"}
        abstract_v, v = self._query_model(entry)
        logger.info(f"{iid}: Predicting for name {name}: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} for variable {name}")
        return v

    def call(self, iid, fct, fct_name, *args, **kwargs):
        entry = {"iid": iid, "name": fct_name, "kind": "call"}
        abstract_v, v = self._query_model(entry)
        logger.info(f"{iid}: Predicting for call: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} as return value of {fct_name}")
        return v

    def attribute(self, iid, base, attr_name):
        entry = {"iid": iid, "name": attr_name, "kind": "attribute"}
        abstract_v, v = self._query_model(entry)
        logger.info(f"{iid}: Predicting for attribute {attr_name}: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} for attribute {attr_name}")
        return v
