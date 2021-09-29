class ValuePredictor:
    def name(self, iid, name):
        if name == "backend":
            v = object()
        elif name == "layers":
            v = []
        elif name == "name":
            v = "abc"
        elif name == "x":
            v = 5
        elif name == "reduction":
            v = 0.5
        else:
            raise Exception(f"{iid}: Predicting for name {name}: ???")
        print(f"{iid}: Predicting for name {name}: {v}")
        return v

    def call(self, iid, fct, *args, **kwargs):
        if iid in [12, 19, 36, 43]:
            v = 5
        else:
            raise Exception(f"{iid}: Predicting for call: ???")
        print(f"{iid}: Predicting for call: {v}")
        return v

    def attribute(self, iid, base, attr_name):
        if attr_name == "image_data_format":
            v = lambda *a, **b: "jpeg"
        elif attr_name == "BatchNormalization":
            v = lambda *a, **b: object()
        elif attr_name == "Activation":
            v = lambda *a, **b: object()
        elif attr_name == "Conv2D":
            v = lambda *a, **b: object()
        elif attr_name == "AveragePooling2D":
            v = lambda *a, **b: object()
        elif attr_name == "int_shape":
            v = lambda *a, **b: [5, 3]
        else:
            raise Exception(
                f"{iid}: Predicting for attribute {attr_name}: ???")
        print(f"{iid}: Predicting for attribute {attr_name}: {v}")
        return v

    def binary_operation(self, iid, left, operator, right):
        raise Exception(
            f"{iid}: Predicting for binary operation left={left}, op={operator}, right={right}: ???")
