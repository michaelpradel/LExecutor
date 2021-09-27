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
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        name_arg = cst.Arg(cst.SimpleString(value=f'"{node.value}"'))
        lambada = cst.Lambda(params=cst.Parameters(), body=updated_node)
        value_arg = cst.Arg(value=lambada)
        call = cst.Call(func=callee_name, args=[iid_arg, name_arg, value_arg])
        return call

    def __create_call_call(self, node, updated_node):
        callee_name = cst.Name(value="_c_")
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        fct_arg = cst.Arg(value=updated_node.func)
        call = cst.Call(func=callee_name, args=[iid_arg,
                                                fct_arg]+list(updated_node.args))
        return call

    def __create_attribute_call(self, node, updated_node):
        callee_name = cst.Name(value="_a_")
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        value_arg = cst.Arg(updated_node.value)
        assert type(node.attr) == cst.Name
        attr_arg = cst.Arg(cst.SimpleString(value=f'"{node.attr.value}"'))
        call = cst.Call(func=callee_name, args=[iid_arg, value_arg, attr_arg])
        return call

    def __create_binop_call(self, node, updated_node):
        callee_name = cst.Name(value="_b_")
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        left_arg = cst.Arg(updated_node.left)
        operator_name = type(node.operator).__name__
        operator_arg = cst.Arg(cst.SimpleString(value=f'"{operator_name}"'))
        right_arg = cst.Arg(updated_node.right)
        call = cst.Call(func=callee_name, args=[
                        iid_arg, left_arg, operator_arg, right_arg])
        return call

    def __create_import(self, name):
        module_name = cst.Name(value="lexecutor_runtime")
        fct_name = cst.Name(value=name)
        imp_alias = cst.ImportAlias(name=fct_name)
        imp = cst.ImportFrom(module=module_name, names=[imp_alias])
        stmt = cst.SimpleStatementLine(body=[imp])
        return stmt

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

    def leave_Attribute(self, node, updated_node):
        wrapped_attribute = self.__create_attribute_call(node, updated_node)
        return wrapped_attribute

    def leave_BinaryOperation(self, node, updated_node):
        wrapped_bin_op = self.__create_binop_call(node, updated_node)
        return wrapped_bin_op
