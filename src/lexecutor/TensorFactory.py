import numpy as np
import torch as t
from .TraceEntries import NameEntry, CallEntry, AttributeEntry, BinOpEntry
from .Hyperparams import Hyperparams as p
from .Util import dtype, device


class TensorFactory(object):
    def __init__(self, token_embedding):
        self.token_embedding = token_embedding

        # for building vocab of values
        self.value_to_index = {}
        self.next_value_index = 0

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
