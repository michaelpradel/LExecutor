import os
import csv
import argparse
import libcst as cst
import pandas as pd
from ..Util import gather_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files or .txt file with all file paths", nargs="+")

class DecoratorRemover(cst.CSTTransformer):
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        # Remove decorators by replacing them with an empty list
        updated_node = updated_node.with_changes(decorators=[])
        return updated_node


if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    
    for file in files:
        with open(file + ".orig", "r") as fp:
            src = fp.read()
        ast = cst.parse_module(src)

        transformer = DecoratorRemover()
        updated_module = ast.visit(transformer)

        # Get the updated code
        updated_code = updated_module.code

        base_dir = file.split('functions_with_invocation')[0]
        file_name = file.split('functions_with_invocation')[1]
        outfile = base_dir + "functions_without_decorator" + file_name

        with open(outfile, "w") as f:
            f.write(updated_code)