import pandas as pd

class NameEntry(object):
    def __init__(self, iid, name, value):
        self.iid = iid
        self.name = name
        self.value = value


class CallEntry(object):
    def __init__(self, iid, fct_name, args, value):
        self.iid = iid
        self.fct_name = fct_name
        self.args = args
        self.value = value


class AttributeEntry(object):
    def __init__(self, iid, base, attr_name, value):
        self.iid = iid
        self.base = base
        self.attr_name = attr_name
        self.value = value


class BinOpEntry(object):
    def __init__(self, iid, left, operator, right, value):
        self.iid = iid
        self.left = left
        self.operator = operator
        self.right = right
        self.value = value


def read_trace(file):
    print(f"Reading trace file {file}")
    entries = []
    name_trace = pd.read_hdf(file, key="name")
    with open(file) as fp:
        for line in fp.readlines():
            segments = line.rstrip().split(" ")
            kind = segments[1]
            if kind == "name":
                entries.append(
                    NameEntry(segments[0], segments[2], segments[3]))
            elif kind == "call":
                entries.append(
                    CallEntry(segments[0], segments[2], segments[3:-1], segments[-1]))
            elif kind == "attribute":
                entries.append(AttributeEntry(
                    segments[0], segments[2], segments[3], segments[4]))
            elif kind == "binary_operation":
                entries.append(BinOpEntry(
                    segments[0], segments[2], segments[3], segments[4], segments[5]))
    print(f"Found {len(entries)} entries in trace file")
    return entries
