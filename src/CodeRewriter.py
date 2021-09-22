import libcst as cst
from libcst.metadata import ParentNodeProvider, PositionProvider


class CodeRewriter(cst.CSTTransformer):

    METADATA_DEPENDENCIES = (ParentNodeProvider, PositionProvider,)

    def __init__(self, file_path, iids, used_names):
        self.file_path = file_path
        self.used_names = used_names
        self.iids = iids

    def __create_iid(self, node):
        location = self.get_metadata(PositionProvider, node)
        line = location.start.line
        column = location.start.column
        iid = self.iids.new(self.file_path, line, column)
        return iid

    def __create_call(self, node, updated_node):
        callee_name = cst.Name(value="_lexecutor_")
        lambada = cst.Lambda(params=cst.Parameters(), body=updated_node)
        value_arg = cst.Arg(value=lambada)
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        call = cst.Call(func=callee_name, args=[value_arg, iid_arg])
        return call

    def __create_import(self):
        module_name = cst.Name(value="lexecutor_runtime")
        fct_name = cst.Name(value="_lexecutor_")
        imp_alias = cst.ImportAlias(name=fct_name)
        imp = cst.ImportFrom(module=module_name, names=[imp_alias])
        stmt = cst.SimpleStatementLine(body=[imp])
        return stmt

    def __is_attribute_use(self, name_node):
        parent = self.get_metadata(ParentNodeProvider, name_node)
        if type(parent) == cst.Attribute:
            # don't instrument right-most attribute on left-hand side of assignment
            # (because that's a definition, not a use)
            grand_parent = self.get_metadata(ParentNodeProvider, parent)
            if type(grand_parent) == cst.AssignTarget:
                return False
            return True
        return False

    # don't visit lines marked with special comment
    def visit_SimpleStatementLine(self, node):
        c = node.trailing_whitespace.comment
        if c is not None and c.value == "# don't instrument":
            print(f"Ignoring line with comment")
            return False
        print(f"Will consider line")
        return True

    # add import of our runtime library to the file
    def leave_Module(self, node, updated_node):
        import_stmt = self.__create_import()
        print(f"Old body's type:\n {type(updated_node.body)}")
        new_body = [import_stmt]+list(updated_node.body)
        return updated_node.with_changes(body=new_body)

    # rewrite Name nodes to intercept values they resolve to
    def leave_Name(self, node, updated_node):
        if node in self.used_names or self.__is_attribute_use(node):
            wrapped_name = self.__create_call(node, updated_node)
            return wrapped_name
        else:
            return updated_node
