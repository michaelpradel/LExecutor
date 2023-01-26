import os
import csv
import argparse
import pandas as pd
from ..Util import gather_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files to count lines or .txt file with all file paths", nargs="+")

def count_lines(file_path):
    total_lines = 0
    with open(file_path, "r") as file:
        lines = file.readlines()
        docstring = False
        for line in lines:
            line = line.strip()
            
            if line == "" \
                or line.startswith("#") \
                or docstring and not (line.startswith('"""') or line.startswith("'''"))\
                or (line.startswith("'''") and line.endswith("'''") and len(line) >3)  \
                or (line.startswith('"""') and line.endswith('"""') and len(line) >3) :
                    continue
            elif line.startswith('"""') or line.startswith("'''"):
                docstring = not docstring
                continue
            else:
                total_lines += 1
    return total_lines

def save_total_lines(files, total_lines):
    # Create CSV file and add header if it doesn't exist
    if not os.path.isfile('./total_lines.csv'):
        columns = ["file", "total_lines"]

        with open('./total_lines.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(columns)
            
    df = pd.read_csv('./total_lines.csv')
    df_new_data = pd.DataFrame({
        'file': files,
        'total_lines': total_lines
    })
    df = pd.concat([df, df_new_data])
    df.to_csv('./total_lines.csv', index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    
    total_lines = []
    for file_path in files:
        total_lines.append(count_lines(f"{file_path}.orig"))
    save_total_lines(files, total_lines)