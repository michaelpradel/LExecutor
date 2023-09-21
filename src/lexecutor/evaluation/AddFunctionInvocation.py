import os
import csv
import argparse
import libcst as cst
import pandas as pd
from ..Util import gather_files

parser = argparse.ArgumentParser()
parser.add_argument(
    "--files", help="Python files or .txt file with all file paths", nargs="+")


class TransformerVisitor(cst.CSTTransformer):
    def __init__(self):
        self.contains_wrapper = False
        self.lines_to_discard = 0

    def leave_ClassDef(self, node, updated_node):
        if updated_node.name.value == "Wrapper":
            self.contains_wrapper = True
        return updated_node

    def leave_FunctionDef(self, node, updated_node):
        self.function_name = updated_node.name.value
        self.function_parameters = updated_node.params.params
        # Remove decorators by replacing them with an empty list
        updated_node = updated_node.with_changes(decorators=[])
        return updated_node

    def leave_Module(self, node, updated_node):
        condition = cst.parse_expression("__name__ == '__main__'")

        function_args = ""
        for parameter in self.function_parameters:
            value = parameter.name.value
            if value != "self":
                function_args += f"{value}, "
        function_args = function_args[:-2]

        if self.contains_wrapper:
            # the function is a method
            if self.function_name != "__init__":
                additional_code_body = [cst.parse_statement("obj = Wrapper()")]
                additional_code_body = additional_code_body + [cst.parse_statement(f"obj.{self.function_name}({function_args})")]
                self.lines_to_discard = 5
            else:
                additional_code_body = [cst.parse_statement(f"Wrapper({function_args})")]
                self.lines_to_discard = 4
        else:
            additional_code_body = [cst.parse_statement(f"{self.function_name}({function_args})")]
            self.lines_to_discard = 3
            
        additional_code = cst.If(
                    test=cst.Comparison(
                        left=cst.Name(
                            value='__name__',
                            lpar=[],
                            rpar=[],
                        ),
                        comparisons=[
                            cst.ComparisonTarget(
                                operator=cst.Equal(
                                    whitespace_before=cst.SimpleWhitespace(
                                        value=' ',
                                    ),
                                    whitespace_after=cst.SimpleWhitespace(
                                        value=' ',
                                    ),
                                ),
                                comparator=cst.SimpleString(
                                    value='"__main__"',
                                    lpar=[],
                                    rpar=[],
                                ),
                            ),
                        ],
                        lpar=[],
                        rpar=[],
                    ),
                    body=cst.IndentedBlock(
                        body=additional_code_body
                    ),
                    orelse=None
                )


        if self.function_name == "__init__":
            pass

        new_body = []
        for i in range(len(updated_node.body)):
            stmt = updated_node.body[i]
            new_body.append(stmt)
        new_body.append(additional_code)

        return updated_node.with_changes(body=new_body)
    
def save_aux_data(dataset_name, file, lines_to_discard):
    # Create CSV file and add header if it doesn't exist
    if not os.path.isfile(f'./aux_data_{dataset_name}.csv'):
        columns = ['file', 'lines_to_discard']

        with open(f'./aux_data_{dataset_name}.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(columns)

    df = pd.read_csv(f'./aux_data_{dataset_name}.csv')
    df_new_data = pd.DataFrame({
        'file': [file],
        'lines_to_discard': [lines_to_discard]
    })
    df = pd.concat([df, df_new_data])
    df.to_csv(f'./aux_data_{dataset_name}.csv', index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    files = gather_files(args.files)
    
    for file in files:
        with open(file, "r") as fp:
            src = fp.read()
        ast = cst.parse_module(src)
        ast_wrapper = cst.metadata.MetadataWrapper(ast)

        code_transformer = TransformerVisitor()

        rewritten_ast = ast_wrapper.visit(code_transformer)
        rewritten_code = rewritten_ast.code
        
        base_dir = file.split('functions')[0]
        file_name = file.split('functions')[1]
        outfile = base_dir + "functions_with_invocation" + file_name

        with open(outfile, "w") as f:
            f.write(rewritten_code)

        save_aux_data("functions_with_invocation_dataset", file, code_transformer.lines_to_discard)