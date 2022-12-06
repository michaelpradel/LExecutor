import argparse
from .Util import gather_files
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
        self.next_id = 0

    def set_source_file(self, file):
        self.file = file

    def leave_Return(self, node, updated_node):
        args = [cst.Arg(value=node.value)] if node.value is not None else []
        expr = cst.Expr(
            value=cst.Call(
                func=cst.Name("exit"),
                args=args
            )
        )
        return expr

    def leave_FunctionDef(self, node, updated_node):
        body = [s for s in updated_node.body.body]
        out_code = cst.Module(body=body).code

        outfile = os.path.join(self.dest_dir, f"body_{self.next_id}.py")
        info = f"# Extracted from {self.file}"
        with open(outfile, "w") as f:
            f.write(info+"\n")
            f.write(out_code)
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
