import argparse
import os
import signal
import subprocess
from ..Util import gather_files


parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files or .txt file with all file paths", nargs="+")
parser.add_argument(
    "--tests", help="Run pytest tests", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    files = gather_files(args.files)
    
    if args.tests:
        command = "pytest"
    else:
        command = "python"

    # run the files (with a timeout)
    for file in files:
        try:
            process = subprocess.Popen(
                f"time {command} {file}", shell=True, start_new_session=True)
            process.wait(timeout=30)  # seconds
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
