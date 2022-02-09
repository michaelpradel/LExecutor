import os
from os import path
from .ValueAbstraction import abstract_value


class TraceWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        if path.exists(self.file_path):
            os.remove(self.file_path)
        self.buffer = []

    def append_name(self, iid, name, raw_value):
        value = abstract_value(raw_value)
        self.__append(f"{iid} name {name} {value}")

    def append_call(self, iid, fct, raw_args, raw_kwargs, raw_value):
        all_raw_args = list(raw_args) + list(raw_kwargs.values())
        args = [abstract_value(r) for r in all_raw_args]
        args = " ".join(args)
        value = abstract_value(raw_value)
        fct_name = fct.__name__ if hasattr(fct, "__name__") else str(fct)
        if " " in fct_name:  # some fcts that don't have a proper name
            fct_name = fct_name.split(" ")[0]
        self.__append(f"{iid} call {fct_name} {args} {value}")

    def append_attribute(self, iid, raw_base, attr_name, raw_value):
        base = abstract_value(raw_base)
        value = abstract_value(raw_value)
        self.__append(f"{iid} attribute {base} {attr_name} {value}")

    def append_binary_operation(self, iid, raw_left, operator, raw_right, raw_value):
        left = abstract_value(raw_left)
        right = abstract_value(raw_right)
        value = abstract_value(raw_value)
        self.__append(
            f"{iid} binary_operation {left} {operator} {right} {value}")

    def __append(self, line):
        self.buffer.append(line)
        if len(self.buffer) % 100000 == 0:
            self.flush()

    def flush(self):
        trace_segment = ""
        for line in self.buffer:
            trace_segment += line + "\n"
        with open(self.file_path, "a") as file:
            file.write(trace_segment)
        self.buffer = []
