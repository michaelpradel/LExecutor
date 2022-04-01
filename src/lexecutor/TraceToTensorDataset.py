from torch.utils.data import IterableDataset
import pandas as pd
from random import shuffle
from .TraceEntries import NameEntry, CallEntry, AttributeEntry, BinOpEntry


class TraceToTensorDataset(IterableDataset):
    def __init__(self, trace_file, tensor_factory):
        # load traces
        self.name_df = pd.read_hdf(trace_file, key="name")
        self.call_df = pd.read_hdf(trace_file, key="call")
        self.attribute_df = pd.read_hdf(trace_file, key="attribute")

        # prepare shuffled list of (frame, index) pairs
        self.frame_index_pairs = []
        for i in self.name_df.index.values:
            self.frame_index_pairs.append((self.name_df, i))
        for i in self.call_df.index.values:
            self.frame_index_pairs.append((self.call_df, i))
        for i in self.attribute_df.index.values:
            self.frame_index_pairs.append((self.attribute_df, i))
        shuffle(self.frame_index_pairs)

        self.tensor_factory = tensor_factory

    def __iter__(self):
        for frame, index in self.frame_index_pairs:
            row = frame.iloc[index]
            if frame is self.name_df:
                entry = NameEntry(row[0], row[1], row[2])
            elif frame is self.call_df:
                args = row[2].split(" ")
                entry = CallEntry(row[0], row[1], args, row[3])
            elif frame is self.attribute_df:
                entry = AttributeEntry(row[0], row[1], row[2], row[3])
            yield self.tensor_factory.entry_to_tensors(entry)

    def __len__(self):
        return len(self.name_df) + len(self.call_df) + len(self.attribute_df)
