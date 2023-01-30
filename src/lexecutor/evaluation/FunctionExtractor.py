import argparse
from ..Util import gather_files
import libcst as cst
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files to extract from or .txt file with all file paths", nargs="+")
parser.add_argument(
    "--dest", help="Destination directory", required=True)


class ExtractorVisitor(cst.CSTTransformer):
    def __init__(self, dest_dir):
        self.dest_dir = dest_dir

        existing_files = [f for f in os.listdir(dest_dir)]
        self.next_id = 0
        while f"body_{self.next_id}.py" in existing_files:
            self.next_id += 1

    def set_source_file(self, file):
        self.file = file

    def leave_Param(self, node, updated_node):
        # remove parameter type annotation
        return updated_node.with_changes(annotation=None)

    def leave_FunctionDef(self, node, updated_node):
        info = f"# Extracted from {self.file}"
        
        if len(updated_node.params.params) and updated_node.params.params[0].name.value == "self":
            # wrap function into a class
            code = cst.Module([]).code_for_node(
                cst.ClassDef(
                    name=cst.Name(
                        value='Wrapper'
                    ),
                    body=cst.IndentedBlock(
                        body=[updated_node.with_changes(returns=None)]
                    )
                )
            )
        else:
            # remove return type annotation and save full function
            code = cst.Module([]).code_for_node(
                updated_node.with_changes(returns=None))
        
        outfile = os.path.join(
            f"{self.dest_dir}/functions", f"function_{self.next_id}.py")
        with open(outfile, "w") as f:
            f.write(info+"\n")
            f.write(code)

        self.next_id += 1

        return updated_node


if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    extractor = ExtractorVisitor(args.dest)
    for file in files:
        with open(file, "r") as fp:
            src = fp.read()
        ast = cst.parse_module(src)
        extractor.set_source_file(file)
        ast.visit(extractor)
