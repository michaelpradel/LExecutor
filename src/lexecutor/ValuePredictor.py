from abc import ABC


class ValuePredictor(ABC):
    def name(self, iid, name):
        pass

    def call(self, iid, fct, *args, **kwargs):
        pass

    def attribute(self, iid, base, attr_name):
        pass

    def binary_operation(self, iid, left, operator, right):
        pass
