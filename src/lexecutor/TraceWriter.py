import pandas as pd
from .ValueAbstraction import abstract_value


class TraceWriter:
    def __init__(self, file_path):
        self.file_path = file_path

        self.name_df = pd.DataFrame(data=None)
        self.name_buffer = []

        self.call_df = pd.DataFrame(data=None)
        self.call_buffer = []

        self.attribute_df = pd.DataFrame(data=None)
        self.attribute_buffer = []

    def __flush_buffer(self, buffer, old_df):
        new_df = pd.DataFrame(data=buffer)
        all_df = pd.concat([old_df, new_df])
        self.name_buffer = []
        return all_df

    def append_name(self, iid, name, raw_value):
        value = abstract_value(raw_value)
        self.name_buffer.append([iid, name, value])

        if len(self.name_buffer) % 100000 == 0:
            self.name_df = self.__flush_buffer(self.name_buffer, self.name_df)

    def append_call(self, iid, fct, raw_args, raw_kwargs, raw_value):
        all_raw_args = list(raw_args) + list(raw_kwargs.values())
        args = [abstract_value(r) for r in all_raw_args]
        args = " ".join(args)
        value = abstract_value(raw_value)
        fct_name = fct.__name__ if hasattr(fct, "__name__") else str(fct)
        if " " in fct_name:  # some fcts that don't have a proper name
            fct_name = fct_name.split(" ")[0]

        self.call_buffer.append([iid, fct_name, args, value])
        if len(self.call_buffer) % 100000 == 0:
            self.call_df = self.__flush_buffer(self.call_buffer, self.call_df)

    def append_attribute(self, iid, raw_base, attr_name, raw_value):
        base = abstract_value(raw_base)
        value = abstract_value(raw_value)

        self.attribute_buffer.append([iid, base, attr_name, value])
        if len(self.attribute_buffer) % 100000 == 0:
            self.attribute_df = self.__flush_buffer(
                self.attribute_buffer, self.attribute_df)

    def write_to_file(self):
        self.name_df = self.__flush_buffer(self.name_buffer, self.name_df)
        self.name_df[2].astype("category")
        self.name_df.to_hdf(self.file_path, key="name",
                            complevel=9, complib="bzip2")

        self.call_df = self.__flush_buffer(self.call_buffer, self.call_df)
        self.call_df[2].astype("category")
        self.call_df[3].astype("category")
        self.call_df.to_hdf(self.file_path, key="call",
                            complevel=9, complib="bzip2")

        self.attribute_df = self.__flush_buffer(
            self.attribute_buffer, self.attribute_df)
        self.attribute_df[1].astype("category")
        self.attribute_df[3].astype("category")
        self.attribute_df.to_hdf(self.file_path, key="attribute",
                                 complevel=9, complib="bzip2")
