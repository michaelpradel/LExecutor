import pandas as pd
import torch
from random import shuffle
from torch.utils.data import IterableDataset
from ..Util import gather_files


class MaskedValueDataset(IterableDataset):
    def __init__(self, trace_files, input_factory):
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
            self.attribute_df = pd.concat(
                [self.attribute_df, current_attribute_df])

        # remove name-value duplicates
        self.name_df['pair_id'] = self.name_df[1].astype(
            str) + self.name_df[2].astype(str)
        self.name_df = self.name_df.drop_duplicates(
            subset='pair_id', keep="first")
        self.name_df = self.name_df.drop(columns=['pair_id'])
        self.name_df = self.name_df.reset_index()

        self.call_df['pair_id'] = self.call_df[1].astype(
            str) + self.call_df[2].astype(str) + self.call_df[3].astype(str)
        self.call_df = self.call_df.drop_duplicates(
            subset='pair_id', keep="first")
        self.call_df = self.call_df.drop(columns=['pair_id'])
        self.call_df = self.call_df.reset_index()

        self.attribute_df['pair_id'] = self.attribute_df[1].astype(
            str) + self.attribute_df[2].astype(str) + self.attribute_df[3].astype(str)
        self.attribute_df = self.attribute_df.drop_duplicates(
            subset='pair_id', keep="first")
        self.attribute_df = self.attribute_df.drop(columns=['pair_id'])
        self.attribute_df = self.attribute_df.reset_index()

        # prepare shuffled list of (frame, index) pairs
        self.frame_index_pairs = []
        for i in self.name_df.index.values:
            self.frame_index_pairs.append((self.name_df, i))
        for i in self.call_df.index.values:
            self.frame_index_pairs.append((self.call_df, i))
        for i in self.attribute_df.index.values:
            self.frame_index_pairs.append((self.attribute_df, i))
        shuffle(self.frame_index_pairs)

        self.input_factory = input_factory

    def __iter__(self):
        for frame, index in self.frame_index_pairs:
            row = frame.iloc[index]
            if frame is self.name_df:
                entry = row
            elif frame is self.call_df:
                # TO DO: Check the root of the problem during data collection and fix it
                if '(' in row[1]:
                    continue
                entry = [row[0], row[1], row[3]]
            elif frame is self.attribute_df:
                entry = [row[0], row[2], row[3]]
            yield self.input_factory.entry_to_inputs(entry)

    def __len__(self):
        return len(self.name_df) + len(self.call_df) + len(self.attribute_df)
