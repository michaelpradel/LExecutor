import argparse
import re
from ..Util import gather_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files to fix indentation or .txt file with all file paths", nargs="+")

def fix_file(file_path):
    new_content = ""
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in range(len(lines)):
            lines[line] = lines[line].replace('\n', '')
        lines.remove('')
        for line in range(len(lines)):
            line_content = lines[line]
            if line_content.startswith("_l_("):
                previous_line_indentation = len(re.findall("^ *", lines[line-1])[0])
                previous_line_indentation_ = len(re.findall("^ *", lines[line-2])[0])
                try:
                    if not lines[line+1]:
                        next_line_indentation = len(re.findall("^ *", lines[line+2])[0])
                    else:
                        next_line_indentation = len(re.findall("^ *", lines[line+1])[0])
                except:
                    next_line_indentation = 0
                if next_line_indentation > previous_line_indentation_:
                    current_line_indentation = " " * next_line_indentation
                elif previous_line_indentation > previous_line_indentation_:
                    current_line_indentation = " " * previous_line_indentation
                else:
                    current_line_indentation = " " * previous_line_indentation_
                line_content = current_line_indentation + line_content
            new_content += line_content + "\n"
                
    with open(file_path, "w") as file:
        file.write(new_content)

if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    
    for file_path in files:
        print(f"Fixing {file_path}")
        fix_file(file_path)