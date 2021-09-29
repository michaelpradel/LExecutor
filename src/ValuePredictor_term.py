class ValuePredictor:
    def name(self, iid, name):
        if name == "nodes":
            v = [1,2,3]
        elif name == "makeTerm":
            v = lambda *a, **b: "foo"
        elif name == "title":
            v = "bar"
        elif name == "term":
            v = "baz"
        else:
            raise Exception(f"{iid}: Predicting for name {name}: ???")
        print(f"{iid}: Predicting for name {name}: {v}")
        return v

    def call(self, iid, fct, *args, **kwargs):
        if iid == 11111:
            v = 5
        else:
            raise Exception(f"{iid}: Predicting for call: ???")
        print(f"{iid}: Predicting for call: {v}")
        return v

    def attribute(self, iid, base, attr_name):
        if attr_name == "xxxx":
            v = lambda *a, **b: "foo"
        else:
            raise Exception(
                f"{iid}: Predicting for attribute {attr_name}: ???")
        print(f"{iid}: Predicting for attribute {attr_name}: {v}")
        return v

    def binary_operation(self, iid, left, operator, right):
        raise Exception(
            f"{iid}: Predicting for binary operation left={left}, op={operator}, right={right}: ???")
