from unicodedata import name
import os
import numpy as np
from .TraceReader import read_trace, NameEntry, CallEntry, AttributeEntry, BinOpEntry
from .Hyperparams import Hyperparams as p


class TensorFactory(object):
    def __init__(self, token_embedding):
        self.token_embedding = Embedding(token_embedding)

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

    def traces_to_tensors(self, trace_paths, npz_dest_dir):
        print(f"Transforming {len(trace_paths)} trace files to tensors")

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
                args = np.zeros((p.max_call_args, p.value_emb_len))
                base = np.zeros(p.value_emb_len)
                left = np.zeros(p.value_emb_len)
                right = np.zeros(p.value_emb_len)
                operator = np.zeros(p.token_emb_len)

                kind = np.zeros(4)
                if type(entry) is NameEntry:
                    kind[0] = 1
                    name = self.token_embedding.get(entry.name)
                elif type(entry) is CallEntry:
                    kind[1] = 1
                    name = self.token_embedding.get(entry.fct_name)
                    for arg_idx, arg in enumerate(entry.args[:p.max_call_args]):
                        args[arg_idx] = self.__value_to_one_hot(arg)
                elif type(entry) is AttributeEntry:
                    kind[2] = 1
                    name = self.token_embedding.get(entry.attr_name)
                    base = self.__value_to_one_hot(entry.base)
                elif type(entry) is BinOpEntry:
                    kind[3] = 1
                    name = np.zeros(p.token_emb_len)
                    left = self.__value_to_one_hot(entry.left)
                    right = self.__value_to_one_hot(entry.right)
                    operator = self.token_embedding.get(entry.operator)

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
