import torch as t
import numpy as np
import json
from .ValuePredictor import ValuePredictor
from .Train import load_FastText
from .Model import ValuePredictionModel
from .TraceEntries import NameEntry, CallEntry, AttributeEntry, BinOpEntry
from .TensorFactory import TensorFactory, Embedding
from .Util import dtype, device
from .Hyperparams import Hyperparams as p
from .ValueAbstraction import restore_value


class NeuralValuePredictor(ValuePredictor):
    def __init__(self):
        p.batch_size = 1

        # load model
        print("Loading value prediction model")
        self.model = ValuePredictionModel()
        self.model.load_state_dict(
            t.load("data/models/default", map_location=device))
        print(f"Loaded model: {self.model}")

        # load embedding
        ft = load_FastText(
            "data/embeddings/default/embedding")
        self.token_embedding = Embedding(ft)

        self.tensor_factory = TensorFactory(ft)

        with open("data/tensors/value_map.json") as f:
            value_to_index = json.load(f)
            self.index_to_value = {i: v for v, i in value_to_index.items()}

    def __query_model(self, entry):
        kind, name, args, base, left, right, operator, _ = self.tensor_factory.entry_to_tensors(
            entry)
        kind = [tensor.cpu() for tensor in kind]
        name = [tensor.cpu() for tensor in name]
        args = [[tensor.cpu() for tensor in arg] for arg in args]
        base = [tensor.cpu() for tensor in base]
        left = [tensor.cpu() for tensor in left]
        right = [tensor.cpu() for tensor in right]
        operator = [tensor.cpu() for tensor in operator]
        xs_kind = t.tensor(np.array([kind]), dtype=dtype, device=device)
        xs_name = t.tensor(np.array([name]), dtype=dtype, device=device)
        xs_args = t.tensor(np.array([args]), dtype=dtype, device=device)
        xs_base = t.tensor(np.array([base]), dtype=dtype, device=device)
        xs_left = t.tensor(np.array([left]), dtype=dtype, device=device)
        xs_right = t.tensor(np.array([right]), dtype=dtype, device=device)
        xs_operator = t.tensor(
            np.array([operator]), dtype=dtype, device=device)

        with t.no_grad():
            self.model.eval()
            pred_ys = self.model((xs_kind, xs_name, xs_args,
                                  xs_base, xs_left, xs_right, xs_operator))
        max_index = t.argmax(pred_ys[0]).item()
        predicted_value = restore_value(self.index_to_value[max_index])
        return predicted_value

    def name(self, iid, name):
        entry = NameEntry(iid, name, None)
        return self.__query_model(entry)

    def call(self, iid, fct, *args, **kwargs):
        entry = CallEntry(iid, fct, args, None)
        return self.__query_model(entry)

    def attribute(self, iid, base, attr_name):
        entry = AttributeEntry(iid, base, attr_name, None)
        return self.__query_model(entry)

    def binary_operation(self, iid, left, operator, right):
        entry = BinOpEntry(iid, left, operator, right, None)
        return self.__query_model(entry)


# for testing
if __name__ == "__main__":
    vp = NeuralValuePredictor()

    v = vp.name(23, "first_name")
    print(f"Predicted value for name 'first_name': {v}")

    v = vp.call(23, "is_list_like", "@object")
    print(
        f"Predicted value for call to 'is_list_like' with '@object' argument: {v}")

    v = vp.attribute(23, "@object", "aslist")
    print(f"Predicted value for attribute 'aslist' of '@object' argument: {v}")

    v = vp.binary_operation(23, "@set", "BitAnd", "@set")
    print(f"Predicted result of binary operation '@set' 'BitAnd' '@set': {v}")
