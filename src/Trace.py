import os
from os import path
from ValueAbstraction import abstract_value


class Trace:
    def __init__(self, file_path):
        self.file_path = file_path
        if path.exists(self.file_path):
            os.remove(self.file_path)
        self.buffer = []

    def append(self, raw_value, iid):
        value = abstract_value(raw_value)
        self.buffer.append((value, iid))
        if len(self.buffer) % 1000 == 0:
            self.flush()

    def flush(self):
        trace_segment = ""
        for value, iid in self.buffer:
            trace_segment += f"{iid} {value}\n"
        with open(self.file_path, "a") as file:
            file.write(trace_segment)
        self.buffer = []
