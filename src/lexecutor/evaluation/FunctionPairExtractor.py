import argparse
from git import Repo
import re
from collections import namedtuple
from os.path import exists, isdir, join
from os import mkdir
import libcst as cst

# Helper script to find commits that modify a single function,
# and to extract the pair of old+new function into separate files.

parser = argparse.ArgumentParser()
parser.add_argument(
    "--repo", help="Directory with a git repository", required=True)
parser.add_argument(
    "--dest", help="Destination directory", required=True)


CodeChange = namedtuple("CodeChange", ["old_commit", "new_commit", "file", "line"])


def find_code_changes(repo):
    commits = list(repo.iter_commits("main"))
    code_changes = []
    for c in commits:
        if len(c.parents) == 0:
            continue
        diff = c.parents[0].diff(c, create_patch=True)
        if len(diff) == 1 and diff[0].a_path and diff[0].a_path.endswith(".py") and diff[0].b_path and diff[0].b_path.endswith(".py"):
            diff_str = str(diff[0])
            matches = re.findall(r"@@", diff_str)
            if len(matches) == 2:
                line_info = diff_str.split("@@")[1]
                line = int(line_info[line_info.find("-")+1:line_info.find(",")])
                code_changes.append(CodeChange(c.parents[0].hexsha, c.hexsha, diff[0].a_path, line))
                # TODO remove after testing
                if len(code_changes) == 5:
                    break
    return code_changes


class FunctionExtractor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, line):
        self.line = line
        self.function = None

    def visit_FunctionDef(self, node):
        start = self.get_metadata(cst.metadata.PositionProvider, node).start
        end = self.get_metadata(cst.metadata.PositionProvider, node).end
        if start.line <= self.line <= end.line:
            self.function = cst.Module([]).code_for_node(node)


def extract_function(file, line):
    tree = cst.parse_module(open(file).read())
    tree = cst.MetadataWrapper(tree)
    extractor = FunctionExtractor(line)
    tree.visit(extractor)
    if extractor.function is None:
        print(f"WARNING: No function found in {file} at line {line}")
        return "MISSING"
    return extractor.function


def extract_function_pair(repo, code_change, dest):
    # get old function
    repo.git.checkout(code_change.old_commit)
    file_path = join(repo.working_tree_dir, code_change.file)
    old_function = extract_function(file_path, code_change.line)

    # get new function
    repo.git.checkout(code_change.new_commit)
    file_path = join(repo.working_tree_dir, code_change.file)
    new_function = extract_function(file_path, code_change.line)

    # write both functions to destination directory
    with open(join(dest, "old.py"), "w") as f:
        f.write(old_function)
    with open(join(dest, "new.py"), "w") as f:
        f.write(new_function)


if __name__ == "__main__":
    args = parser.parse_args()
    if not exists(args.repo) or not isdir(args.repo):
        print(f"Invalid repo directory: {args.repo}")
        exit(1)
    if not exists(args.dest) or not isdir(args.dest):
        print(f"Invalid destination directory: {args.dest}")
        exit(1)    

    repo = Repo(args.repo)
    code_changes = find_code_changes(repo)
    print(f"{len(code_changes)} code changes found")
    for code_change_idx, code_change in enumerate(code_changes):
        dest_dir = join(args.dest, f"code_change_{code_change_idx}")
        mkdir(dest_dir)
        extract_function_pair(repo, code_change, dest_dir)