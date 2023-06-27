import argparse
import os
import signal
import subprocess
from ..Util import gather_files
from ..Hyperparams import Hyperparams as params

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files or .txt file with all file paths", nargs="+")
parser.add_argument(
    "--tests", help="Run pytest tests", action="store_true")
parser.add_argument(
    "--log_dest_dir", help="Destination directory for the log files", required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    files = gather_files(args.files)
    
    if args.tests:
        command = "pytest"
    else:
        command = "python3"

    # run the files (with a timeout)
    for file in files:
        project_name = file.split("/")[2]
        for execution in range(1, params.number_executions+1):
            if params.dataset == "random_functions":
                file_name = file.split("/")[4].split('.')[0]
            else:
                file_name = file.split("/")[2].split('.')[0]

            log_file = open(f"{args.log_dest_dir}/{project_name}_{file_name}_{str(execution)}.txt", "w")
            try:
                process = subprocess.Popen(
                    f"time {command} {file} {execution}", shell=True, start_new_session=True, stdout=log_file, stderr=log_file)
                process.wait(timeout=30)  # seconds
            except subprocess.TimeoutExpired:
                log_file.write("TimeLimit!!!!")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
