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