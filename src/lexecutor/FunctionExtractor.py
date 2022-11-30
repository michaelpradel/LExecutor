import argparse
import libcst
from .Util import gather_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files or .txt file with all file paths to extract functions", nargs="+")


class FunctionExtractorTransformer(libcst.CSTTransformer):

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> libcst.CSTNode:
        code = f"{libcst.Module([]).code_for_node(updated_node.with_changes(params=libcst.Parameters()))}\n\nif __name__ == '__main__':\n\t{updated_node.name.value}()"

        with open(f"./functions_under_test/{updated_node.name.value}.py", "w") as file:
            file.write(code)

        return updated_node


if __name__ == "__main__":
    args = parser.parse_args()

    files = gather_files(args.files)

    for file_path in files:
        with open(file_path, "r") as file:
            src = file.read()

            ast = libcst.parse_module(src)
            ast.visit(FunctionExtractorTransformer())
