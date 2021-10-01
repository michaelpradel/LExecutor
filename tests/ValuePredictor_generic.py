class Toy:
    pass


class ValuePredictor:
    def name(self, iid, name):
        v = Toy()
        print(f"{iid}: Predicting for name {name}: {v}")
        return v

    def call(self, iid, fct, *args, **kwargs):
        v = Toy()
        print(f"{iid}: Predicting for call: {v}")
        return v

    def attribute(self, iid, base, attr_name):
        v = Toy()
        print(f"{iid}: Predicting for attribute {attr_name}: {v}")
        return v

    def binary_operation(self, iid, left, operator, right):
        v = 3
        print(f"{iid}: Predicting result of {operator} operation: {v}")
        return v
