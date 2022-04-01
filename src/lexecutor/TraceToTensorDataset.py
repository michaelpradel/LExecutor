from torch.utils.data import IterableDataset
import pandas as pd
from .TraceReader import NameEntry, CallEntry, AttributeEntry, BinOpEntry


class TraceToTensorDataset(IterableDataset):
    def __init__(self, trace_file, tensor_factory):
        self.df = pd.read_hdf(trace_file, key="name")
        self.tensor_factory = tensor_factory

    def __iter__(self):
        for _, row in self.df.iterrows():
            entry = NameEntry(row[0], row[1], row[2])
            yield self.tensor_factory.entry_to_tensors(entry)

    def __len__(self):
        return len(self.df)
