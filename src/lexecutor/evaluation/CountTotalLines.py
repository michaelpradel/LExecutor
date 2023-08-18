import os
import csv
import argparse
import pandas as pd
from ..Util import gather_files
from ..Hyperparams import Hyperparams as params

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files to count lines or .txt file with all file paths", nargs="+")

def count_lines(file_path):
    total_lines = 0
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("_l_("):
                total_lines += 1
    return total_lines

def save_total_lines(files, total_lines):
    # Create CSV file and add header if it doesn't exist
    if not os.path.isfile(f'./total_lines_{params.dataset}_dataset.csv'):
        columns = ["file", "total_lines"]

        with open(f'./total_lines_{params.dataset}_dataset.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(columns)
            
    df = pd.read_csv(f'./total_lines_{params.dataset}_dataset.csv')
    df_new_data = pd.DataFrame({
        'file': files,
        'total_lines': total_lines
    })
    df = pd.concat([df, df_new_data])
    df.to_csv(f'./total_lines_{params.dataset}_dataset.csv', index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    
    total_lines = []
    for file_path in files:
        total_lines.append(count_lines(f"{file_path}"))
    save_total_lines(files, total_lines)