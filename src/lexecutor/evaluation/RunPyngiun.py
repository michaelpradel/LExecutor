import argparse
import os
import subprocess
from ..Util import gather_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files with extracted functions or .txt file with all file paths", nargs="+")
parser.add_argument(
    "--dest", help="Destination directory for the tests", required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    os.environ['PYNGUIN_DANGER_AWARE'] = 'x'
    
    pynguin_parameters = '--maximum_search_time 30 --seed 42 --max-attempts 10 --maximum_test_execution_timeout 10 --maximum_slicing_time 10 --test_execution_time_per_statement 1 --assertion-generation SIMPLE -v'

    for file in files:
        if file.startswith(os.getcwd()):
            file = file[len(os.getcwd())+1:]
        module_pynguin_path = file.replace("/", ".")[2:-3]
        print(f"Running Pynguin on {module_pynguin_path} with parameters {pynguin_parameters}")
        log_pynguin = subprocess.run(f"pynguin --project-path . --output-path {args.dest} --module-name {module_pynguin_path} {pynguin_parameters}".split(
            " "), capture_output=True, text=True, shell=False, timeout=60)
