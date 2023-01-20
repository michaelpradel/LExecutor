from ..ValuePredictor import ValuePredictor
from ...Util import device
from .ModelServer import ModelServer
from ...Logging import logger
import time
import requests
from requests.exceptions import ConnectionError
import subprocess
from ...ValueAbstraction import restore_value


class CodeT5ValuePredictor(ValuePredictor):
    def __init__(self, stats):
        self.stats = stats

    def _query_model(self, entry):
        def get(entry):
            raw_response = requests.get(
                "http://localhost:5000/query", params=entry)
            if raw_response.status_code != 200:
                raise RuntimeError(
                    f"Model server returned error code {raw_response.status_code}")
            return raw_response.json()

        response = None
        try:
            response = get(entry)
        except ConnectionError:
            # model server not yet running; start it
            logger.info("No model server running. Starting it now")
            server_log = open("model_server.log", "w")
            subprocess.Popen(
                "python -m lexecutor.predictors.codet5.ModelServer".split(" "),
                stderr=server_log, stdout=server_log)

            # try to connect until it's responding (or we give up)
            attempts = 0
            while attempts < 5:
                try:
                    response = get(entry)
                    logger.info("Model server is up and running")
                    break
                except ConnectionError:
                    time.sleep(5)  # seconds
                    attempts += 1

        if response is None:
            raise RuntimeError("Could not connect to model server")

        val_as_string = response["v"]
        val = restore_value(val_as_string)

        return val_as_string, val

    def name(self, iid, name):
        entry = {"iid": iid, "name": name, "kind": "name"}
        abstract_v, v = self._query_model(entry)
        logger.info(f"{iid}: Predicting for name {name}: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} for variable {name}")
        return v

    def call(self, iid, fct, fct_name, *args, **kwargs):
        entry = {"iid": iid, "name": fct_name, "kind": "call"}
        abstract_v, v = self._query_model(entry)
        logger.info(f"{iid}: Predicting for call: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} as return value of {fct_name}")
        return v

    def attribute(self, iid, base, attr_name):
        entry = {"iid": iid, "name": attr_name, "kind": "attribute"}
        abstract_v, v = self._query_model(entry)
        logger.info(f"{iid}: Predicting for attribute {attr_name}: {v}")
        self.stats.inject_value(
            iid, f"Inject {abstract_v} for attribute {attr_name}")
        return v
