import os
import csv
import argparse
import pandas as pd
from ..Util import gather_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files to get wrapp info or .txt file with all file paths", nargs="+")

def get_wrapp_info(file_path):
    wrapped = 0
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("class Wrapper:"):
                wrapped = 1
                break
    return wrapped

def save_wrapp_info(files, wrapp_info):
    # Create CSV file and add header if it doesn't exist
    if not os.path.isfile('./wrapp_info.csv'):
        columns = ["file", "wrapped"]

        with open('./wrapp_info.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(columns)

    df = pd.read_csv('./wrapp_info.csv')
    df_new_data = pd.DataFrame({
        'file': files,
        'wrapped': wrapp_info
    })
    df = pd.concat([df, df_new_data])
    df.to_csv('./wrapp_info.csv', index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)

    wrapp_info = []
    for file_path in files:
        wrapp_info.append(get_wrapp_info(f"{file_path}"))
    save_wrapp_info(files, wrapp_info)
