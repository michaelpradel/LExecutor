import pandas as pd
import os
import csv
from .Logging import logger


class RuntimeStats:
    def __init__(self, iids):
        self.iids = iids
        self.total_uses = 0
        self.guided_uses = 0

        self.covered_iids = set()

        self.event_trace = []

    def cover_iid(self, iid):
        self.covered_iids.add(iid)
        self.event_trace.append(f"Line {self.iids.line(iid)}: Executed")

    def inject_value(self, iid, msg):
        self.event_trace.append(
            f"Line {self.iids.line(iid)}: {msg}")

    def uncaught_exception(self, iid, e):
        self.event_trace.append(
            f"Line {self.iids.line(iid)}: Uncaught exception {type(e)}\n{e}")

    def print(self):
        logger.info(f"Covered iids: {len(self.covered_iids)}")
        logger.info(f"Total uses: {self.total_uses}")
        logger.info(f"Guided uses : {self.guided_uses}/{self.total_uses}")

    def _save_summary_metrics(self, file, predictor_name):
        # Create CSV file and add header if it doesn't exist
        if not os.path.isfile('./metrics.csv'):
            columns = ['file', 'predictor', 'covered_iids',
                       'total_uses', 'guided_uses']

            with open('./metrics.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(columns)

        df = pd.read_csv('./metrics.csv')
        df_new_data = pd.DataFrame({
            'file': [file],
            'predictor': [predictor_name],
            'covered_iids': [len(self.covered_iids)],
            'total_uses': [self.total_uses],
            'guided_uses': [f"{self.guided_uses}/{self.total_uses}"]
        })
        df = pd.concat([df, df_new_data])
        df.to_csv('./metrics.csv', index=False)

    def _save_event_trace(self):
        with open("trace.txt", "w") as fp:
            fp.write("\n".join(self.event_trace))

    def save(self, file, predictor_name):
        self._save_summary_metrics(file, predictor_name)
        self._save_event_trace()
