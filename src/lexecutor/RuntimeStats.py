import pandas as pd
import os
from os import path
import csv
import time
from .Logging import logger
from .IIDs import IIDs
from .Hyperparams import Hyperparams as param

write_event_trace = True
write_metrics = True


class RuntimeStats:
    def __init__(self, execution):
        self.total_uses = 0
        self.guided_uses = 0

        self.covered_iids = set()
        self.executed_lines = []

        if write_event_trace:
            self.event_trace = []
            self.iids = IIDs(param.iids_file)

        self.random_predictions = 0
        self.type4py_predictions = 0

        self.execution = execution

    def cover_iid(self, iid):
        self.covered_iids.add(iid)
        if write_event_trace:
            self.event_trace.append(f"Line {self.iids.line(iid)}: Executed")
            
    def cover_line(self, iid):
        self.executed_lines.append(iid)
        logger.info(f"Line {self.iids.line(iid)}: Executed")

    def inject_value(self, iid, msg):
        if write_event_trace:
            self.event_trace.append(
                f"Line {self.iids.line(iid)}: {msg}")

    def uncaught_exception(self, iid, e):
        if write_event_trace:
            self.event_trace.append(
                f"Line {self.iids.line(iid)}: Uncaught exception {type(e)}\n{e}")

    def print(self):
        logger.info(f"Covered iids: {len(self.covered_iids)}")
        logger.info(f"Total uses: {self.total_uses}")
        logger.info(f"Guided uses : {self.guided_uses}/{self.total_uses}")

    def _save_summary_metrics(self, file, predictor_name, execution_time):
        if write_metrics:
            if param.dataset == "so_snippets":
                project_name = ""
                file_name = file.split("/")[2].split('.')[0]
            else:
                project_name = file.split("/")[2]
                file_name = file.split("/")[4].split('.')[0]

            # Create CSV file and add header if it doesn't exist
            if not os.path.isfile(f'./metrics_{project_name}_{file_name}_{self.execution}.csv'):
                columns = ['file', 'predictor', 'covered_iids',
                        'total_uses', 'guided_uses', 'executed_lines', 
                        'covered_lines', 'execution_time', 'random_predictions', 
                        'type4py_predictions', 'execution']

                with open(f'./metrics_{project_name}_{file_name}_{self.execution}.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(columns)

            df = pd.read_csv(f'./metrics_{project_name}_{file_name}_{self.execution}.csv')
            df_new_data = pd.DataFrame({
                'file': [file],
                'predictor': [predictor_name],
                'covered_iids': [len(self.covered_iids)],
                'total_uses': [self.total_uses],
                'guided_uses': [self.guided_uses],
                'executed_lines': [len(self.executed_lines)],
                'covered_lines': [len(set(self.executed_lines))],
                'execution_time': [execution_time],
                'random_predictions': [self.random_predictions],
                'type4py_predictions': [self.type4py_predictions],
                'execution': [self.execution]
            })
            df = pd.concat([df, df_new_data])
            df.to_csv(f'./metrics_{project_name}_{file_name}_{self.execution}.csv', index=False)

    def _save_event_trace(self):
        with open("trace.txt", "w") as fp:
            fp.write("\n".join(self.event_trace))

    def save(self, file, predictor_name, start_time):
        self._save_summary_metrics(file, predictor_name, time.time() - start_time)
        if write_event_trace:
            self._save_event_trace()
