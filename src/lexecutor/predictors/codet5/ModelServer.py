from pathlib import Path
import torch as t
import numpy as np
from flask import Flask, json, request
import requests
from ...Util import device
from ...Hyperparams import Hyperparams as params
from ...IIDs import IIDs
from .FineTune import load_CodeT5
from .InputFactory import InputFactory
from ...Logging import logger
import logging

# TODO auto-kill the server after some time of inactivity


class ModelServer:
    def __init__(self):
        self._initialize_model()
        self._initialize_http_server()

    def _fetch_model(self, model_path):
        path_to_url = {
            "data/released_models/codet5_model_20230105_fine-grained.bin": "https://github.com/michaelpradel/LExecutor/releases/download/Models_20230105/codet5_model_20230105_fine-grained.bin",
            "data/released_models/codet5_model_20230105_coarse-grained.bin": "https://github.com/michaelpradel/LExecutor/releases/download/Models_20230105/codet5_model_20230105_coarse-grained.bin"
        }
        if Path(model_path).exists():
            return
        Path.mkdir(Path(model_path).parent, parents=True, exist_ok=True)
        logger.info(f"Downloading model from {path_to_url[model_path]}")
        request = requests.get(path_to_url[model_path], allow_redirects=True)
        open(model_path, 'wb').write(request.content)        

    def _initialize_model(self):
        logger.info("Loading CodeT5 model")
        self.tokenizer, self.model = load_CodeT5()

        if params.value_abstraction == "fine-grained":
            model_path = "data/released_models/codet5_model_20230105_fine-grained.bin"
        elif params.value_abstraction == "coarse-grained-deterministic" or params.value_abstraction == "coarse-grained-randomized":
            model_path = "data/released_models/codet5_model_20230105_coarse-grained.bin"
        self._fetch_model(model_path)
        self.model.load_state_dict(t.load(model_path, map_location=device))

        iids = IIDs(params.iids_file)
        self.input_factory = InputFactory(iids, self.tokenizer)
        logger.info("CodeT5 model loaded")

    def _initialize_http_server(self):
        logger.info("Starting HTTP server")
        api = Flask(__name__)
        flask_log = logging.getLogger('werkzeug')
        flask_log.setLevel(logging.ERROR)

        @api.route('/query', methods=['GET'])
        def handle_query():
            # reconstruct entry from REST API request
            entry = {"iid": request.args.get("iid"),
                     "name": request.args.get("name"),
                     "kind": request.args.get("kind")}

            # turn query into vectors
            input_ids, _ = self.input_factory.entry_to_inputs(entry)
            input_ids = [tensor.cpu() for tensor in input_ids]

            # query the model and decode the result
            with t.no_grad():
                self.model.eval()
                generated_ids = self.model.generate(
                    t.tensor(np.array([input_ids]), device=device), max_length=params.max_output_length)

            predicted_value = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True)

            if params.verbose:
                if self.tokenizer.bos_token_id not in generated_ids or self.tokenizer.eos_token_id not in generated_ids[0]:
                    print(
                        f"Warning: CodeT5 likely produced a garbage value: {predicted_value}")

            # respond with a JSON object
            result = {"v": predicted_value}
            return json.dumps(result)

        api.run()



if __name__ == "__main__":
    ModelServer()
