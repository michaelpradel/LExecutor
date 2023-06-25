import argparse
from ..Util import gather_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files or .txt file with all file paths", nargs="+")

def remove_last_line(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove the last line
    if lines:
        lines = lines[:-1]

    # Save the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

    print(f"The last line has been removed from {file_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)

    for file in files:
        remove_last_line(file)