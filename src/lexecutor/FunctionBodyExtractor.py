import argparse
from .Util import gather_files
import libcst as cst
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files to extract from or .txt file with all file paths", nargs="+")
parser.add_argument(
    "--dest", help="Destination directory")

fut_name = "LExecutor_function_under_test"

class ExtractorVisitor(cst.CSTTransformer):
    def __init__(self, dest_dir):
        self.dest_dir = dest_dir
        self.next_id = 0

    def visit_FunctionDef(self, node):
        out_tree = cst.Module(
            body=[
                cst.FunctionDef(
                    name=cst.Name(fut_name),
                    params=cst.Parameters(),
                    body=node.body
                ),
                cst.If(
                    test=cst.Comparison(
                        left=cst.Name(value="__name__"),
                        comparisons=[
                            cst.ComparisonTarget(
                                operator=cst.Equal(),
                                comparator=cst.SimpleString(value='"__main__"')
                            )
                        ]
                    ),
                    body=cst.IndentedBlock(
                        body=[
                            cst.SimpleStatementLine(
                                body=[
                                    cst.Expr(
                                        value=cst.Call(
                                            func=cst.Name(fut_name)
                                        )
                                    )
                                ]
                            )
                        ]
                    )
                )
            ]
        )

        outfile = os.path.join(self.dest_dir, f"body_{self.next_id}.py")
        with open(outfile, "w") as f:
            f.write(out_tree.code)
        self.next_id += 1


if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    extractor = ExtractorVisitor(args.dest)
    for file in files:
        with open(file, "r") as file:
            src = file.read()
        ast = cst.parse_module(src)
        ast.visit(extractor)
