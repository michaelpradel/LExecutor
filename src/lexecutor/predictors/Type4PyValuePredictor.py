from .RandomPredictor import RandomPredictor
from ..Logging import logger
from ..ValueAbstraction import restore_value
from ..IIDs import IIDs
import requests

class Type4PyValuePredictor(RandomPredictor):
    def __init__(self, code_snippet_file, stats):
        super().__init__()
        self.type_predictions = self._query_model(code_snippet_file)
        print(self.type_predictions)
        self.stats = stats
        
    def _query_model(self, code_snippet_file):
        with open(code_snippet_file + '.orig') as file:
            raw_response = requests.post(
                "http://localhost:5001/api/predict?tc=0", file.read())
        if raw_response.status_code != 200:
            raise RuntimeError(
                f"Model server returned error code {raw_response.status_code}")
        return raw_response.json()

    def _get_abstract_value(self, name):
        abstract_value = None
        predicted_type = False # boolean aux var

        if self.type_predictions["response"]:
            # global var
            if name in self.type_predictions["response"]["variables"]:
                abstract_value = self.type_predictions["response"]["variables_p"][name][0][0].split('[')[0].lower()
                predicted_type = True
            else:
                # in function
                functions = self.type_predictions["response"]["funcs"]
                if self.type_predictions["response"]["classes"]:
                    for class_ in self.type_predictions["response"]["classes"]:
                        functions += class_["funcs"]
    
                for fct in functions:
                    # variables
                    if name in fct["variables"]:
                        if fct["variables_p"][name] and fct["variables_p"][name][0]:
                            abstract_value = fct["variables_p"][name][0][0].split('[')[0].lower()
                            predicted_type = True
                            break
                    # parameters
                    elif name in fct["params"]:
                        if fct["params_p"][name] and fct["params_p"][name][0]:
                            abstract_value = fct["params_p"][name][0][0].split('[')[0].lower()
                            predicted_type = True
                            break
                    # return
                    elif "ret_type_p" in fct and name == fct["name"]:
                        if fct["ret_type_p"] and fct["ret_type_p"][0]:
                            abstract_value = fct["ret_type_p"][0][0].split('[')[0].lower()
                            predicted_type = True
                            break

        return abstract_value, predicted_type
    
    def name(self, iid, name):
        abstract_v, predicted_type = self._get_abstract_value(name)
        if predicted_type:
            self.stats.type4py_predictions += 1
            v = restore_value(abstract_v)
            logger.info(f"{iid}: Predicting with Type4Py for name {name}: {v}")
            return v
        else:
            self.stats.random_predictions += 1
            super().name(iid, name)

    def call(self, iid, fct, fct_name, *args, **kwargs):
        abstract_v, predicted_type = self._get_abstract_value(fct_name)
        if predicted_type:
            self.stats.type4py_predictions += 1
            v = restore_value(abstract_v)
            logger.info(f"{iid}: Predicting with Type4Py for call {fct_name}: {v}")
            return v
        else:
            self.stats.random_predictions += 1
            super().call(iid, fct, fct_name, *args, **kwargs)

    def attribute(self, iid, base, attr_name):
        abstract_v, predicted_type = self._get_abstract_value(attr_name)
        if predicted_type:
            self.stats.type4py_predictions += 1
            v = restore_value(abstract_v)
            logger.info(f"{iid}: Predicting with Type4Py for attribute {attr_name}: {v}")
            return v
        else:
            self.stats.random_predictions += 1
            super().attribute(iid, base, attr_name)
