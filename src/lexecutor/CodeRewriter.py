import libcst as cst
from libcst.metadata import ParentNodeProvider, PositionProvider


class CodeRewriter(cst.CSTTransformer):

    METADATA_DEPENDENCIES = (ParentNodeProvider, PositionProvider,)

    ignored_names = ["True", "False", "None"]
    ignored_calls = ["super"]  # special function names to not instrument

    def __init__(self, file_path, iids, used_names):
        self.file_path = file_path
        self.used_names = used_names
        self.iids = iids

        self.quotation_char = '"'  # flipped to "'" when inside an f-string

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

        # create a lambda like "lambda a=a: a" to avoid refering to undefined class variables
        param = cst.Param(name=updated_node, default=updated_node)

        name_arg = cst.Arg(cst.SimpleString(
            value=f"{self.quotation_char}{node.value}{self.quotation_char}"))
        lambada = cst.Lambda(params=cst.Parameters(
            params=[param]), body=updated_node)
        value_arg = cst.Arg(value=lambada)
        call = cst.Call(func=callee_name, args=[iid_arg, name_arg, value_arg])
        return call

    def __ensure_generator_expr_have_parens(self, args):
        # make sure that generator expressions have parentheses if not the only argument
        updated_args = []
        for arg in args:
            if (isinstance(arg.value, cst.GeneratorExp)
                    and len(arg.value.lpar) == 0
                    and len(arg.value.rpar) == 0):
                g = arg.value
                g_new = cst.GeneratorExp(elt=g.elt,
                                         for_in=g.for_in,
                                         lpar=[cst.LeftParen()],
                                         rpar=[cst.RightParen()])
                updated_args.append(cst.Arg(value=g_new))
            else:
                updated_args.append(arg)
        return updated_args

    def __create_call_call(self, node, updated_node):
        callee_name = cst.Name(value="_c_")
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        fct_arg = cst.Arg(value=updated_node.func)
        all_args = [iid_arg, fct_arg] + \
            self.__ensure_generator_expr_have_parens(updated_node.args)
        call = cst.Call(func=callee_name, args=all_args)
        return call

    def __create_attribute_call(self, node, updated_node):
        callee_name = cst.Name(value="_a_")
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        value_arg = cst.Arg(updated_node.value)
        assert type(node.attr) == cst.Name
        attr_arg = cst.Arg(cst.SimpleString(
            value=f"{self.quotation_char}{node.attr.value}{self.quotation_char}"))
        call = cst.Call(func=callee_name, args=[iid_arg, value_arg, attr_arg])
        return call

    def __create_binop_call(self, node, updated_node):
        callee_name = cst.Name(value="_b_")
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        left_arg = cst.Arg(updated_node.left)
        operator_name = type(node.operator).__name__
        operator_arg = cst.Arg(cst.SimpleString(
            value=f"{self.quotation_char}{operator_name}{self.quotation_char}"))
        right_arg = cst.Arg(updated_node.right)
        call = cst.Call(func=callee_name, args=[
                        iid_arg, left_arg, operator_arg, right_arg])
        return call

    def __create_import(self, name):
        module_name = cst.Attribute(value=cst.Name(
            value="lexecutor"), attr=cst.Name(value="runtime"))
        fct_name = cst.Name(value=name)
        imp_alias = cst.ImportAlias(name=fct_name)
        imp = cst.ImportFrom(module=module_name, names=[imp_alias])
        stmt = cst.SimpleStatementLine(body=[imp])
        return stmt

    def __wrap_import(self, node, updated_node):
        try_stmt = cst.Try(body=cst.IndentedBlock(
            body=[cst.SimpleStatementLine(body=[updated_node])]),
            handlers=[cst.ExceptHandler(body=cst.IndentedBlock(
                body=[cst.SimpleStatementLine(body=[cst.Pass()])]),
                type=cst.Name(value="ImportError"))])
        return try_stmt

    def __is_l_value(self, node):
        parent = self.get_metadata(ParentNodeProvider, node)
        return type(parent) == cst.AssignTarget

    def __is_ignored_call(self, call_node):
        if type(call_node.func) == cst.Name:
            return call_node.func.value in self.ignored_calls
        else:
            return False

    def visit_SimpleStatementLine(self, node):
        # don't visit lines marked with special comment
        c = node.trailing_whitespace.comment
        if c is not None and c.value == "# don't instrument":
            print(f"Ignoring line with comment")
            return False
        return True

    def visit_Import(self, node):
        # don't instrument imports, as we'll wrap them in try-except
        return False

    def visit_ImportFrom(self, node):
        # don't instrument imports, as we'll wrap them in try-except
        return False

    def visit_FormattedString(self, node):
        if node.start == 'f"':
            self.quotation_char = "'"
        elif node.start == "f'":
            self.quotation_char = '"'
        return True

    def leave_Call(self, node, updated_node):
        # rewrite Call nodes to intercept function calls
        if not self.__is_ignored_call(node):
            wrapped_call = self.__create_call_call(node, updated_node)
            return wrapped_call
        else:
            return updated_node

    def leave_Name(self, node, updated_node):
        # rewrite Name nodes to intercept values they resolve to
        if node in self.used_names and node.value not in self.ignored_names:
            wrapped_name = self.__create_name_call(node, updated_node)
            return wrapped_name
        else:
            return updated_node

    def leave_Attribute(self, node, updated_node):
        if self.__is_l_value(node):
            return updated_node
        wrapped_attribute = self.__create_attribute_call(node, updated_node)
        return wrapped_attribute

    def leave_BinaryOperation(self, node, updated_node):
        wrapped_bin_op = self.__create_binop_call(node, updated_node)
        return wrapped_bin_op

    def leave_SimpleStatementLine(self, node, updated_node):
        # surround imports with try-except;
        # cannot do this in leave_Import because we need to replace the import's parent node
        if isinstance(node.body[0], cst.Import) or isinstance(node.body[0], cst.ImportFrom):
            if not (isinstance(node.body[0], cst.ImportFrom) and node.body[0].module.value == "__future__"):
                wrapped_import = self.__wrap_import(
                    node.body[0], updated_node.body[0])
                return wrapped_import
        return updated_node

    def leave_Module(self, node, updated_node):
        # check for "__future__" imports; they must remain at beginning of file
        target_idx = 0  # index to add our imports at
        for i in range(len(updated_node.body)):
            stmt = updated_node.body[i]
            if (isinstance(stmt, cst.SimpleStatementLine)
               and isinstance(stmt.body[0], cst.ImportFrom)
               and stmt.body[0].module.value == "__future__"):
                target_idx = i + 1

        # add our imports
        import_n = self.__create_import("_n_")
        import_a = self.__create_import("_a_")
        import_c = self.__create_import("_c_")
        import_b = self.__create_import("_b_")

        new_body = (list(updated_node.body[:target_idx])
                    + [import_n, import_a, import_c, import_b]
                    + list(updated_node.body[target_idx:]))

        return updated_node.with_changes(body=new_body)
