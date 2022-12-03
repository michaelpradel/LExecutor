import pandas as pd
import os
import csv


class RuntimeStats:
    total_uses = 0
    guided_uses = 0

    covered_iids = set()

    def cover_iid(self, iid):
        self.covered_iids.add(iid)

    def print(self):
        print(f"Covered iids: {len(self.covered_iids)}")
        print(f"Total uses: {self.total_uses}")
        print(f"Guided uses : {self.guided_uses}/{self.total_uses}")

    def save(self, file, predictor_name):
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
