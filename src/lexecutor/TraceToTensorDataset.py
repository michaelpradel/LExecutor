from torch.utils.data import IterableDataset
import pandas as pd
from random import shuffle
from .TraceEntries import NameEntry, CallEntry, AttributeEntry
from .Util import gather_files


class TraceToTensorDataset(IterableDataset):
    def __init__(self, trace_files, tensor_factory):
        # load traces
        self.name_df = pd.DataFrame(data=None)
        self.call_df = pd.DataFrame(data=None)
        self.attribute_df = pd.DataFrame(data=None)

        trace_files = gather_files(trace_files)
        for trace_file in trace_files:
            try:
                current_name_df = pd.read_hdf(trace_file, key="name")
            except:
                current_name_df = pd.DataFrame(data=None)

            try:
                current_call_df = pd.read_hdf(trace_file, key="call")
            except:
                current_call_df = pd.DataFrame(data=None)

            try:
                current_attribute_df = pd.read_hdf(trace_file, key="attribute")
            except:
                current_attribute_df = pd.DataFrame(data=None)

            self.name_df = pd.concat([self.name_df, current_name_df])
            self.call_df = pd.concat([self.call_df, current_call_df])
            self.attribute_df = pd.concat([self.attribute_df, current_attribute_df])

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
