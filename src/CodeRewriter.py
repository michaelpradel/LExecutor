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

    def __create_name_call(self, node, updated_node):
        callee_name = cst.Name(value="_n_")
        lambada = cst.Lambda(params=cst.Parameters(), body=updated_node)
        value_arg = cst.Arg(value=lambada)
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        call = cst.Call(func=callee_name, args=[value_arg, iid_arg])
        return call

    def __create_call_call(self, node, updated_node):
        callee_name = cst.Name(value="_c_")
        # TODO: iid
        fct_arg = cst.Arg(value=updated_node.func)
        call = cst.Call(func=callee_name, args=[
                        fct_arg]+list(updated_node.args))
        return call

    def __create_import(self, name):
        module_name = cst.Name(value="lexecutor_runtime")
        fct_name = cst.Name(value=name)
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
        import_n = self.__create_import("_n_")
        import_a = self.__create_import("_a_")
        import_c = self.__create_import("_c_")
        import_b = self.__create_import("_b_")
        new_body = [import_n, import_a, import_c,
                    import_b]+list(updated_node.body)
        return updated_node.with_changes(body=new_body)

    # rewrite Call nodes to intercept function calls
    def leave_Call(self, node, updated_node):
        wrapped_call = self.__create_call_call(node, updated_node)
        return wrapped_call

    # rewrite Name nodes to intercept values they resolve to
    def leave_Name(self, node, updated_node):
        if node in self.used_names:
            wrapped_name = self.__create_name_call(node, updated_node)
            return wrapped_name
        else:
            return updated_node
