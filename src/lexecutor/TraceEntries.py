import pandas as pd
from typing import List


class NameEntry(object):
    def __init__(self, iid, name, value):
        self.iid = iid
        self.name = name
        self.value = value


class CallEntry(object):
    def __init__(self, iid, fct_name, args: List[str], value):
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

def read_traces(trace_files):
    print(f"Reading trace file {trace_files}")
    entries = []

    files = []
    with open(trace_files) as fp:
        for line in fp.readlines():
            files.append(line.rstrip())

    name_df = pd.DataFrame(data=None)
    call_df = pd.DataFrame(data=None)
    attribute_df = pd.DataFrame(data=None)

    for trace_file in files:
        current_name_df = pd.read_hdf(trace_file, key="name")
        current_call_df = pd.read_hdf(trace_file, key="call")
        current_attribute_df = pd.read_hdf(trace_file, key="attribute")

        name_df = pd.concat([name_df, current_name_df])
        call_df = pd.concat([call_df, current_call_df])
        attribute_df = pd.concat([attribute_df, current_attribute_df])

    frame_index_pairs = []
    for i in name_df.index.values:
        frame_index_pairs.append((name_df, i))
    for i in call_df.index.values:
        frame_index_pairs.append((call_df, i))
    for i in attribute_df.index.values:
        frame_index_pairs.append((attribute_df, i))

    for frame, index in frame_index_pairs:
        row = frame.iloc[index]
        if frame is name_df:
            entry = NameEntry(row[0], row[1], row[2])
        elif frame is call_df:
            args = row[2].split(" ")
            entry = CallEntry(row[0], row[1], args, row[3])
        elif frame is attribute_df:
            entry = AttributeEntry(row[0], row[1], row[2], row[3])
        entries.append(entry)

    return entries
