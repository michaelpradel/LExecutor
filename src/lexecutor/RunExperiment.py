import argparse
import subprocess
from .Util import gather_files


parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files or .txt file with all file paths", nargs="+")

if __name__ == "__main__":
    args = parser.parse_args()

    files = gather_files(args.files)

    # run the files and save stats
    for file in files:
        process = subprocess.Popen(f"python {file}", shell=True)
        process.wait()
