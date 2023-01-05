from collections import namedtuple
import os
from os import path
import json
from .Logging import logger


Location = namedtuple(
    "Location", ["file", "line", "column_start", "column_end"])


class IIDs:
    def __init__(self, file_path):
        if not path.exists(file_path):
            logger.info(f"Creating new iid file at {file_path}")
            self.next_iid = 1
            self._iid_to_location = {}
        else:
            with open(file_path, "r") as file:
                json_object = json.load(file)
            self.next_iid = json_object["next_iid"]
            self._iid_to_location = json_object["iid_to_location"]
        self.file_path = file_path

    def new(self, file, line, column_start, column_end):
        self._iid_to_location[self.next_iid] = Location(
            file, line, column_start, column_end)
        self.next_iid += 1
        return self.next_iid - 1

    def store(self):
        all_data = {
            "next_iid": self.next_iid,
            "iid_to_location": self._iid_to_location,
        }
        json_object = json.dumps(all_data, indent=2)
        with open(self.file_path, "w") as file:
            file.write(json_object)

    def line(self, iid):
        return self._iid_to_location[str(iid)][1]

    def location(self, iid):
        return Location(*self._iid_to_location[str(iid)])
