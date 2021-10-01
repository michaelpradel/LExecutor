import argparse
from os import path
import libcst as cst
from CodeRewriter import CodeRewriter
from IIDs import IIDs
import re
import json


parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files to instrument", nargs="+")
parser.add_argument(
    "--iids", help="JSON file with instruction IDs (will create iids.json if nothing given)")


def gather_accessed_names(ast_wrapper):
    scopes = set(ast_wrapper.resolve(cst.metadata.ScopeProvider).values())
    ranges = ast_wrapper.resolve(cst.metadata.PositionProvider)
    used_names = set()
    for scope in scopes:
        print(f"\nScope: {type(scope)}")
        for access in scope.accesses:
            print(
                f"Access to {access.node.value} at {ranges[access.node].start.line}:{ranges[access.node].start.column}")
            name = access.node
            if name.value not in ("self",):
                used_names.add(name)
    return used_names


def instrument_file(file_path, iids):
    with open(file_path, "r") as file:
        src = file.read()

    ast = cst.parse_module(src)
    ast_wrapper = cst.metadata.MetadataWrapper(ast)
    accessed_names = gather_accessed_names(ast_wrapper)

    code_rewriter = CodeRewriter(file_path, iids, accessed_names)
    rewritten_ast = ast_wrapper.visit(code_rewriter)
    print(f"\n{rewritten_ast.code}")

    rewritten_path = re.sub(r"\.py$", "_instr.py", file_path)
    with open(rewritten_path, "w") as file:
        file.write(rewritten_ast.code)


if __name__ == "__main__":
    args = parser.parse_args()
    iids = IIDs(args.iids)
    for file_path in args.files:
        instrument_file(file_path, iids)
    iids.store()
