from abc import ABC
import torch as t
from .Train import load_FastText

class ValuePredictor(ABC):
    def __init__(self):
        # load model
        t.load("data/models/default")

        # load embedding
        self.token_embedding = load_FastText("data/embeddings/default/embedding")

    def name(self, iid, name):
        pass

    def call(self, iid, fct, *args, **kwargs):
        pass

    def attribute(self, iid, base, attr_name):
        pass

    def binary_operation(self, iid, left, operator, right):
        pass
