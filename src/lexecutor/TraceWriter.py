import pandas as pd
from .ValueAbstraction import abstract_value
from .Logging import logger
from .Util import timestamp


column_names = ["iid", "name", "value", "kind"]


class TraceWriter:
    def __init__(self):
        self.df = pd.DataFrame(columns=column_names)
        self.buffer = []

    def _flush_buffer(self):
        new_df = pd.DataFrame(data=self.buffer, columns=column_names)
        self.df = pd.concat([self.df, new_df])
        self.buffer = []

    def _append(self, iid, name, raw_value, kind):
        value = abstract_value(raw_value)
        self.buffer.append([iid, name, value, kind])

        if len(self.buffer) % 100000 == 0:
            self.write_to_file()

    def append_name(self, iid, name, raw_value):
        self._append(iid, name, raw_value, "name")

    def append_call(self, iid, fct, raw_args, raw_kwargs, raw_value):
        fct_name = fct.__name__ if hasattr(fct, "__name__") else str(fct)
        if " " in fct_name:  # some fcts don't have a proper name
            fct_name = fct_name.split(" ")[0]

        self._append(iid, fct_name, raw_value, "call")

    def append_attribute(self, iid, raw_base, attr_name, raw_value):
        self._append(iid, attr_name, raw_value, "attribute")

    def write_to_file(self):
        file_name = f"trace_{timestamp()}.h5"
        logger.info(f"Flushing buffer and writing to {file_name}")
        self._flush_buffer()
        self.df["iid"] = self.df["iid"].astype("int")
        self.df["name"] = self.df["name"].astype("str")
        self.df["value"] = self.df["value"].astype("str")
        self.df["kind"] = self.df["kind"].astype("str")
        self.df.to_hdf(file_name, key="entries", complevel=9, complib="bzip2")
