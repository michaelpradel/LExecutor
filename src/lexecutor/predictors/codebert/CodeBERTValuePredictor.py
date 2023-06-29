import torch as t
import numpy as np
from ..ValuePredictor import ValuePredictor
from ..DLUtil import device
from .CodeBERT import load_CodeBERT
from .InputFactory import InputFactory
from ...Logging import logger
from transformers import pipeline
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
            model_path = "data/codeBERT_models/fine_grained/pytorch_model_epoch9.bin"
        elif params.value_abstraction == "coarse-grained-deterministic" or params.value_abstraction == "coarse-grained-randomized":
            model_path = "data/codeBERT_models/course_grained/pytorch_model_epoch9.bin"
        self.model.load_state_dict(t.load(
            model_path, map_location=device))
        self.model.to(device)
        logger.info("CodeBERT model loaded")

        self.iids = IIDs(params.iids_file)
        self.stats = stats
        self.input_factory = InputFactory(self.iids, self.tokenizer)

    def _query_model(self, entry):
        # turn entry into vectors
        input_ids, _ = self.input_factory.entry_to_inputs(entry)
        input_ids = [tensor.to(device) for tensor in input_ids]

        # query the model and decode the result
        with t.no_grad():
            self.model.eval()

            fill_mask = pipeline('fill-mask', model=self.model, tokenizer=self.tokenizer, device=0, framework="pt")

            # This is required because the fill-mask pipeline adds special tokens during encoding.
            # If use skip_special_tokens=True, <mask> is discarded as well
            INPUT = self.tokenizer.decode(input_ids)
            INPUT = INPUT.replace("</s>", "")
            INPUT = INPUT.replace("<s>", "")
            INPUT = INPUT.replace("<pad>", "")

            predictions = fill_mask(INPUT)

        val_as_string = predictions[0]['token_str']
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
