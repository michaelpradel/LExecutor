import torch as t
import numpy as np
from ..ValuePredictor import ValuePredictor
from ...Util import device
from .FineTune import load_CodeT5
from .InputFactory import InputFactory
from ...ValueAbstraction import restore_value
from ...Hyperparams import Hyperparams as params


class CodeT5ValuePredictor(ValuePredictor):
    def __init__(self, iids, stats, verbose=False):
        # load model
        self.tokenizer, self.model = load_CodeT5()
        self.model.load_state_dict(t.load(
            "data/codeT5_models/dec16_5_projects/checkpoint-last/pytorch_model.bin", map_location=device))

        self.iids = iids
        self.stats = stats
        self.input_factory = InputFactory(self.iids, self.tokenizer)

        self.verbose = verbose

    def _query_model(self, entry):
        input_ids, _ = self.input_factory.entry_to_inputs(entry)
        input_ids = [tensor.cpu() for tensor in input_ids]

        with t.no_grad():
            self.model.eval()
            generated_ids = self.model.generate(
                t.tensor(np.array([input_ids]), device=device), max_length=params.max_output_length)

        predicted_value = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True)

        if self.verbose:
            if self.tokenizer.bos_token_id not in generated_ids or self.tokenizer.eos_token_id not in generated_ids[0]:
                print(
                    f"Warning: CodeT5 likely produced a garbage value: {predicted_value}")

        return predicted_value, restore_value(predicted_value)

    def name(self, iid, name):
        entry = {"iid": iid, "name": name, "kind": "name"}
        abstract_v, v = self._query_model(entry)
        print(f"{iid}: Predicting for name {name}: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} for variable {name}")
        return v

    def call(self, iid, fct, *args, **kwargs):
        fct_name = fct.__name__ if hasattr(fct, "__name__") else str(fct)
        if " " in fct_name:  # some fcts that don't have a proper name
            fct_name = fct_name.split(" ")[0]
        entry = {"iid": iid, "name": fct_name, "kind": "call"}
        abstract_v, v = self._query_model(entry)
        print(f"{iid}: Predicting for call: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} as return value of {fct_name}")
        return v

    def attribute(self, iid, base, attr_name):
        entry = {"iid": iid, "name": attr_name, "kind": "attribute"}
        abstract_v, v = self._query_model(entry)
        print(f"{iid}: Predicting for attribute {attr_name}: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} for attribute {attr_name}")
        return v
