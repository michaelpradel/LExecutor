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

    def leave_Return(self, node, updated_node):
        args = []
        if node.value is not None:
            if type(node.value) is cst.Tuple:
                args = [cst.Arg(value=cst.Tuple(elements=node.value.elements))]
            else:
                args = [cst.Arg(value=node.value)]
        expr = cst.Expr(
            value=cst.Call(
                func=cst.Name("exit"),
                args=args
            )
        )
        return expr

    def leave_Yield(self, node, updated_node):
        args = []
        if node.value is not None:
            if type(node.value) is cst.Tuple:
                args = [cst.Arg(value=cst.Tuple(elements=node.value.elements))]
            elif type(node.value) is cst.From:
                args = [cst.Arg(value=node.value.item)]
            else:
                args = [cst.Arg(value=node.value)]
        expr = cst.Expr(
            value=cst.Call(
                func=cst.Name("exit"),
                args=args
            )
        )
        return expr

    def leave_FunctionDef(self, node, updated_node):
        info = f"# Extracted from {self.file}"

        # save function body
        body = [s for s in updated_node.body.body]
        body_code = cst.Module(body=body).code
        outfile = os.path.join(
            f"{self.dest_dir}/bodies", f"body_{self.next_id}.py")
        with open(outfile, "w") as f:
            f.write(info+"\n")
            f.write(body_code)

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
