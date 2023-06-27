import argparse
from git import Repo
import re
from collections import namedtuple
from os.path import exists, isdir, join
from os import mkdir
import libcst as cst
from typing import Optional

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
                try:
                    line_info = diff_str.split("@@")[1]
                    line = int(line_info[line_info.find("-")+1:line_info.find(",")])
                    code_changes.append(CodeChange(c.parents[0].hexsha, c.hexsha, diff[0].a_path, line))
                except:
                    print(f"Error parsing diff for commit {c.hexsha} -- ignoring")
                if len(code_changes) == 1000:
                    break
    return code_changes


class FunctionExtractor(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def __init__(self, line):
        self.line = line
        self.function = None
        self.is_method = False
        self.param_names = []

    def leave_Param(self, node, updated_node):
        # remove parameter type annotation
        return updated_node.with_changes(annotation=None)

    def leave_FunctionDef(self, node, updated_node):
        if node.name.value == "__init__":
            # ignore constructors, because we want to compare return values
            return updated_node

        start = self.get_metadata(cst.metadata.PositionProvider, node).start
        end = self.get_metadata(cst.metadata.PositionProvider, node).end
        if start.line <= self.line <= end.line:
            self.function = updated_node.with_changes(returns=None)

            if len(node.params.params) > 0 and node.params.params[0].name.value == "self":
                self.is_method = True

            for param in node.params.params:
                self.param_names.append(param.name.value)

        return updated_node


def extract_function(file, line) -> Optional[cst.FunctionDef]:
    tree = cst.parse_module(open(file).read())
    tree = cst.MetadataWrapper(tree)
    extractor = FunctionExtractor(line)
    tree.visit(extractor)
    return extractor


def write_function_to_file(fct, dest_dir, name_prefix, code_change):
    fct_code = cst.Module([]).code_for_node(fct)
    file_name = join(dest_dir, f"{name_prefix}.py")
    comment = f"# {code_change.old_commit} -- {code_change.new_commit} -- {code_change.file} -- {code_change.line}\n\n"
    all_code = comment + fct_code
    with open(file_name, "w") as f:
        f.write(fct_code)


def create_class_wrapper(fct_node, wrapper_name):
    fct_def_code = cst.Module([]).code_for_node(
        cst.ClassDef(
            name=cst.Name(
                value=wrapper_name
            ),
            body = cst.IndentedBlock([fct_node])
            # body=cst.IndentedBlock(
            #     body=[node.with_changes(=None)]
            # )
        )
    )
    return fct_def_code


def write_function_comparison_script(old_fct_extractor, new_fct_extractor, dest_dir, code_change):
    # create code that defines the functions/methods
    assert old_fct_extractor.is_method == new_fct_extractor.is_method
    if old_fct_extractor.is_method:
        # wrap function into a class
        old_fct_def_code = create_class_wrapper(old_fct_extractor.function, "Wrapper1")
        new_fct_def_code = create_class_wrapper(new_fct_extractor.function, "Wrapper2")
        fct_def_code = old_fct_def_code + "\n\n" + new_fct_def_code
    else:
        # change name of functions to distinguish old and new
        renamed_old_fct = old_fct_extractor.function.with_changes(name=cst.Name(value=old_fct_extractor.function.name.value + "_1"))
        renamed_new_fct = new_fct_extractor.function.with_changes(name=cst.Name(value=new_fct_extractor.function.name.value + "_2"))
        fct_def_code = cst.Module([]).code_for_node(renamed_old_fct) + "\n\n" + cst.Module([]).code_for_node(renamed_new_fct)

    # create code that calls and compares the two functions/methods
    main_code_template = """
if __name__ == "__main__":
    import pathlib
    p = str(pathlib.Path(__file__).parent.resolve())

    try:
        val1 = INVOCATION1
        val2 = INVOCATION2
    except Exception as e:
        print(p + ": Function(s) raised an exception: " + str(type(e)) + " -- " + str(e))
    else:
        if val1 == val2:
            print(p + ": Both functions returned the same value" + str(val1))
        else:
            print(p + ": Functions returned different values: " + str(val1) + " vs. " + str(val2))
    """

    if old_fct_extractor.is_method:
        main_code_template = main_code_template.replace("INVOCATION1", "Wrapper1()." + old_fct_extractor.function.name.value + "(" + ", ".join(old_fct_extractor.param_names[1:]) + ")")
        main_code_template = main_code_template.replace("INVOCATION2", "Wrapper2()." + new_fct_extractor.function.name.value + "(" + ", ".join(new_fct_extractor.param_names[1:]) + ")")
    else:
        main_code_template = main_code_template.replace("INVOCATION1", old_fct_extractor.function.name.value + "_1(" + ", ".join(old_fct_extractor.param_names) + ")")
        main_code_template = main_code_template.replace("INVOCATION2", new_fct_extractor.function.name.value + "_2(" + ", ".join(new_fct_extractor.param_names) + ")")

    comment = f"# {code_change.old_commit} -- {code_change.new_commit} -- {code_change.file} -- {code_change.line}\n\n"

    all_code = comment + fct_def_code + "\n\n" + main_code_template
    file_name = join(dest_dir, "compare.py")
    with open(file_name, "w") as f:
        f.write(all_code)


def extract_function_pair(repo, code_change, dest_dir):
    # get old function
    repo.git.checkout(code_change.old_commit)
    file_path = join(repo.working_tree_dir, code_change.file)
    old_function_extractor = extract_function(file_path, code_change.line)

    # get new function
    repo.git.checkout(code_change.new_commit)
    file_path = join(repo.working_tree_dir, code_change.file)
    new_function_extractor = extract_function(file_path, code_change.line)

    if old_function_extractor.function is None or new_function_extractor.function is None:
        return

    if old_function_extractor.is_method != new_function_extractor.is_method:
        return

    # write original functions into files
    write_function_to_file(old_function_extractor.function, dest_dir, "old", code_change)
    write_function_to_file(new_function_extractor.function, dest_dir, "new", code_change)

    # write both functions to a single file that invokes and compares them
    write_function_comparison_script(old_function_extractor, new_function_extractor, dest_dir, code_change)

    print(f"Extracted function pair to {dest_dir}")


if __name__ == "__main__":
    args = parser.parse_args()
    if not exists(args.repo) or not isdir(args.repo):
        print(f"Invalid repo directory: {args.repo}")
        exit(1)
    if exists(args.dest) and not isdir(args.dest):
        print(f"Destination must be a directory: {args.dest}")
        exit(1)
    if not exists(args.dest):
        mkdir(args.dest)    

    repo = Repo(args.repo)
    code_changes = find_code_changes(repo)
    print(f"{len(code_changes)} code changes found")
    for code_change_idx, code_change in enumerate(code_changes):
        dest_dir = join(args.dest, f"code_change_{code_change_idx}")
        if not exists(dest_dir):
            mkdir(dest_dir)

        try:
            extract_function_pair(repo, code_change, dest_dir)
        except:
            print(f"Something went wrong when extracting from code change {code_change_idx} -- ignoring")