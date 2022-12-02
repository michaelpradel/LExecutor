import torch as t
import numpy as np
from ..ValuePredictor import ValuePredictor
from ...Util import device
from .FineTune import load_CodeT5
from .InputFactory import InputFactory
from ...ValueAbstraction import restore_value
from ...IIDs import IIDs


class CodeT5ValuePredictor(ValuePredictor):
    def __init__(self):

        # load model
        self.tokenizer, self.model = load_CodeT5()
        self.model.load_state_dict(t.load(
            "data/codeT5_models/checkpoint-last/pytorch_model_epoch_1_without_name-value-duplicates.bin", map_location=device))

        self.iids = IIDs('iids_original.json')

        self.input_factory = InputFactory(self.iids, self.tokenizer)

    def __query_model(self, entry):
        input_ids, _ = self.input_factory.entry_to_inputs(entry)
        input_ids = [tensor.cpu() for tensor in input_ids]

        with t.no_grad():
            self.model.eval()
            generated_ids = self.model.generate(
                t.tensor(np.array([input_ids]), device=device), max_length=7)

        predicted_value = self.tokenizer.decode(
            generated_ids[0][5], skip_special_tokens=True)
        return restore_value(predicted_value)

    def name(self, iid, name):
        entry = {"iid": iid, "name": name}
        v = self.__query_model(entry)
        print(f"{iid}: Predicting for name {name}: {v}")
        return v

    def call(self, iid, fct, *args, **kwargs):
        fct_name = fct.__name__ if hasattr(fct, "__name__") else str(fct)
        if " " in fct_name:  # some fcts that don't have a proper name
            fct_name = fct_name.split(" ")[0]
        entry = {"iid": iid, "name": fct_name}
        v = self.__query_model(entry)
        print(f"{iid}: Predicting for call: {v}")
        return v

    def attribute(self, iid, base, attr_name):
        entry = {"iid": iid, "name": attr_name}
        v = self.__query_model(entry)
        print(f"{iid}: Predicting for attribute {attr_name}: {v}")
        return v