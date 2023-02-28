import argparse
from os import path
import libcst as cst
from .CodeRewriter import CodeRewriter
from .IIDs import IIDs
from .Util import gather_files
import re
from shutil import copyfile, move


parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files to instrument or .txt file with all file paths", nargs="+")
parser.add_argument(
    "--iids", help="JSON file with instruction IDs", default="iids.json")
parser.add_argument(
    "--restore", help="Restores uninstrumented files from .py.orig files", action="store_true")
parser.add_argument(
    "--line_coverage_instrumentation", help="Instruments files to calculate line coverage", action="store_true")
parser.add_argument(
    "--verbose", help="Print details, e.g., about exceptions during instrumentation", action="store_true")


ignored_file_suffixes = [
    "ansible/utils/collection_loader/_collection_finder.py",
    "ansible/constants.py",
    "django/db/models/expressions.py"
]


def gather_accessed_names(ast_wrapper):
    scopes = set(ast_wrapper.resolve(cst.metadata.ScopeProvider).values())
    ranges = ast_wrapper.resolve(cst.metadata.PositionProvider)
    used_names = set()
    for scope in scopes:
        for access in scope.accesses:
            name = access.node

            # check for reads of class variables defined in the same class
            # (we cannot wrap them into a lambda)
            if isinstance(scope, cst.metadata.ClassScope) and (all(ref.scope == scope for ref in access.referents)):
                continue

            used_names.add(name)

    return used_names


def instrument_file(file_path, iids, line_coverage_instrumentation):
    for suffix in ignored_file_suffixes:
        if file_path.endswith(suffix):
            print(f"{file_path} is on blacklist -- skipping it")
            return

    with open(file_path, "r") as file:
        src = file.read()

    if "LExecutor: DO NOT INSTRUMENT" in src:
        print(f"{file_path} is already instrumented -- skipping it")
        return

    ast = cst.parse_module(src)
    ast_wrapper = cst.metadata.MetadataWrapper(ast)
    accessed_names = gather_accessed_names(ast_wrapper)

    code_rewriter = CodeRewriter(file_path, iids, line_coverage_instrumentation, accessed_names)
    rewritten_ast = ast_wrapper.visit(code_rewriter)
    # print(f"\n{rewritten_ast.code}")

    copied_file_path = re.sub(r"\.py$", ".py.orig", file_path)
    copyfile(file_path, copied_file_path)

    rewritten_code = "# LExecutor: DO NOT INSTRUMENT\n\n" + rewritten_ast.code
    with open(file_path, "w") as file:
        file.write(rewritten_code)


def restore_file(file_path):
    orig_file_path = re.sub(r"\.py$", ".py.orig", file_path)
    if path.isfile(orig_file_path):
        move(orig_file_path, file_path)
        return True
    else:
        return False


if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    if not args.restore:
        print(f"Found {len(files)} file(s) to instrument")
        iids = IIDs(args.iids)
        for file_path in files:
            try:
                print(f"Instrumenting {file_path}")
                instrument_file(file_path, iids, args.line_coverage_instrumentation)
            except Exception as e:
                print(f"Error while instrumenting {file_path}. Ignoring this file.")
                if args.verbose:
                    print(e)
        iids.store()
    else:
        nb_restored = 0
        for file_path in files:
            if restore_file(file_path):
                nb_restored += 1
        print(f"Have restored {nb_restored} out of {len(files)} file(s)")
