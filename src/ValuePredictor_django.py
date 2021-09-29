class Toy:
    pass


class ValuePredictor:
    def name(self, iid, name):
        if name == "attrs":
            v = {}
        elif name == "__new__":
            v = lambda *a, **b: Toy()
        elif name == "mcs":
            v = [1, 2, 3]
        elif name == "name":
            v = "foo"
        elif name == "bases":
            v = [1, 2, 3]
        elif name == "new_class":
            v = Toy()
        elif name == "a":
            v = Toy()
        elif name == "declared_fields":
            v = ["a", "b"]
        else:
            raise Exception(f"{iid}: Predicting for name {name}: ???")
        print(f"{iid}: Predicting for name {name}: {v}")
        return v

    def call(self, iid, fct, *args, **kwargs):
        if iid in [18, 24]:
            v = Toy()
        else:
            raise Exception(f"{iid}: Predicting for call: ???")
        print(f"{iid}: Predicting for call: {v}")
        return v

    def attribute(self, iid, base, attr_name):
        if attr_name == "__mro__":
            v = [1, 2]
        elif attr_name == "__dict__":
            v = {}
        else:
            raise Exception(
                f"{iid}: Predicting for attribute {attr_name}: ???")
        print(f"{iid}: Predicting for attribute {attr_name}: {v}")
        return v

    def binary_operation(self, iid, left, operator, right):
        if operator == "Subtract":
            v = 3
        else:
            raise Exception(
                f"{iid}: Predicting for binary operation left={left}, op={operator}, right={right}: ???")
        print(f"{iid}: Predicting result of {operator} operation: {v}")
        return v
