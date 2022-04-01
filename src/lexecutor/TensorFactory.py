from unicodedata import name
import os
import json
import numpy as np
import torch as t
from torch.utils.data import IterableDataset, Dataset
from .TraceReader import read_trace, NameEntry, CallEntry, AttributeEntry, BinOpEntry
from .Hyperparams import Hyperparams as p
from .Util import dtype, device


class TensorFactory(object):
    def __init__(self, token_embedding):
        self.token_embedding = token_embedding

        # for building vocab of values
        self.value_to_index = {}
        self.next_value_index = 0

        # for storing in multiple .npz files
        self.next_npz_idx = 0

    def __value_to_one_hot(self, value: str):
        v = np.zeros(p.value_emb_len)
        if value in self.value_to_index:
            v[self.value_to_index[value]] = 1
        else:
            # out-of-bounds exception here means we need to
            #  - increase Hyperprams.value_emb_len or
            #  - use stronger abstraction in ValueAbstraction.abstract_value()
            v[self.next_value_index] = 1
            self.value_to_index[value] = self.next_value_index
            self.next_value_index += 1
        return v

    def __store_tensors(self, dest_dir, xs_kind, xs_name, xs_args, xs_base, xs_left, xs_right, xs_operator, ys_value):
        npz_path = os.path.join(dest_dir, f"{self.next_npz_idx}.npz")
        self.next_npz_idx += 1
        print(f"Storing tensors to {npz_path}")
        np.savez(npz_path, xs_kind=xs_kind, xs_name=xs_name, xs_args=xs_args, xs_base=xs_base,
                 xs_left=xs_left, xs_right=xs_right, xs_operator=xs_operator, ys_value=ys_value)

    def __store_value_map(self, dest_dir):
        with open(os.path.join(dest_dir, "value_map.json"), "w") as f:
            json.dump(self.value_to_index, f)

    def entry_to_tensors(self, entry):
        # x consists of
        #  - kind (name/call/attr/binOp)
        #  - name of var, fct, or attr
        #  - args of call
        #  - base of attr
        #  - left operand
        #  - right operand
        #  - operator
        args = np.zeros((p.max_call_args, p.value_emb_len))
        base = np.zeros(p.value_emb_len)
        left = np.zeros(p.value_emb_len)
        right = np.zeros(p.value_emb_len)
        operator = np.zeros(p.token_emb_len)

        kind = np.zeros(4)
        if type(entry) is NameEntry:
            kind[0] = 1
            name = self.token_embedding.wv[entry.name]
        elif type(entry) is CallEntry:
            kind[1] = 1
            name = self.token_embedding.wv[entry.fct_name]
            for arg_idx, arg in enumerate(entry.args[:p.max_call_args]):
                args[arg_idx] = self.__value_to_one_hot(arg)
        elif type(entry) is AttributeEntry:
            kind[2] = 1
            name = self.token_embedding.wv[entry.attr_name]
            base = self.__value_to_one_hot(entry.base)
        elif type(entry) is BinOpEntry:
            kind[3] = 1
            name = np.zeros(p.token_emb_len)
            left = self.__value_to_one_hot(entry.left)
            right = self.__value_to_one_hot(entry.right)
            operator = self.token_embedding.wv[entry.operator]

        # y consists of
        #  - value
        value = self.__value_to_one_hot(entry.value)

        # move to target device (TODO: merge with the above?)
        x_kind = t.tensor(kind, dtype=dtype, device=device)
        x_name = t.tensor(name, dtype=dtype, device=device)
        x_args = t.tensor(args, dtype=dtype, device=device)
        x_base = t.tensor(base, dtype=dtype, device=device)
        x_left = t.tensor(left, dtype=dtype, device=device)
        x_right = t.tensor(right, dtype=dtype, device=device)
        x_operator = t.tensor(operator, dtype=dtype, device=device)
        y_value = t.tensor(value, dtype=dtype, device=device)

        return x_kind, x_name, x_args, x_base, x_left, x_right, x_operator, y_value

    def traces_to_tensors(self, trace_paths, token_embedding, npz_dest_dir):
        print(f"Transforming {len(trace_paths)} trace files to tensors")

        token_embedding = Embedding(token_embedding)

        # x consists of
        #  - kind (name/call/attr/binOp)
        #  - name of var, fct, or attr
        #  - args of call
        #  - base of attr
        #  - left operand
        #  - right operand
        #  - operator
        xs_kind = []
        xs_name = []
        xs_args = []
        xs_base = []
        xs_left = []
        xs_right = []
        xs_operator = []

        # y consists of
        #  - value
        ys_value = []

        for file_path in trace_paths:
            entries = read_trace(file_path)

            for entry in entries:
                kind, name, args, base, left, right, operator = self.entry_to_tensors(
                    entry, token_embedding)

                xs_kind.append(kind)
                xs_name.append(name)
                xs_args.append(args)
                xs_base.append(base)
                xs_left.append(left)
                xs_right.append(right)
                xs_operator.append(operator)
                ys_value.append(self.__value_to_one_hot(entry.value))

                if len(ys_value) == 100000:
                    self.__store_tensors(
                        npz_dest_dir, xs_kind, xs_name, xs_args, xs_base, xs_left, xs_right, xs_operator, ys_value)
                    xs_kind = []
                    xs_name = []
                    xs_args = []
                    xs_base = []
                    xs_left = []
                    xs_right = []
                    xs_operator = []
                    ys_value = []

        self.__store_tensors(npz_dest_dir, xs_kind, xs_name, xs_args,
                             xs_base, xs_left, xs_right, xs_operator, ys_value)
        self.__store_value_map(npz_dest_dir)

    def tensors_as_dataset(self, npz_dir):
        return DiskDataset(npz_dir)


class Embedding():
    def __init__(self, embedding):
        self.embedding = embedding
        self.cache = {}

    def get(self, token):
        if token in self.cache:
            return self.cache[token]
        else:
            vec = self.embedding.wv[token]
            self.cache[token] = vec
            return vec


class DiskDataset(IterableDataset):
    def __init__(self, npz_dir):
        self.files = [os.path.join(npz_dir, f)
                      for f in os.listdir(npz_dir) if f.endswith(".npz")]

    def __iter__(self):
        for f in self.files:
            with np.load(f) as data:
                xs_kind = t.tensor(data['xs_kind'], dtype=dtype, device=device)
                xs_name = t.tensor(data['xs_name'], dtype=dtype, device=device)
                xs_args = t.tensor(data['xs_args'], dtype=dtype, device=device)
                xs_base = t.tensor(data['xs_base'], dtype=dtype, device=device)
                xs_left = t.tensor(data['xs_left'], dtype=dtype, device=device)
                xs_right = t.tensor(
                    data['xs_right'], dtype=dtype, device=device)
                xs_operator = t.tensor(
                    data['xs_operator'], dtype=dtype, device=device)
                ys_value = t.tensor(
                    data['ys_value'], dtype=dtype, device=device)

                for i in range(xs_kind.shape[0]):
                    yield xs_kind[i], xs_name[i], xs_args[i], xs_base[i], xs_left[i], xs_right[i], xs_operator[i], ys_value[i]
