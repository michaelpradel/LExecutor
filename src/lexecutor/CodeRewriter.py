import libcst as cst
from libcst.metadata import ParentNodeProvider, PositionProvider


class CodeRewriter(cst.CSTTransformer):

    METADATA_DEPENDENCIES = (ParentNodeProvider, PositionProvider,)

    ignored_names = ["True", "False", "None"]
    ignored_calls = ["super"]  # special function names to not instrument

    def __init__(self, file_path, iids, line_coverage_instrumentation, used_names):
        self.file_path = file_path
        self.used_names = used_names
        self.iids = iids
        self.line_coverage_instrumentation = line_coverage_instrumentation

        self.instrument = True  # turned off in special cases, e.g., inside nested f-strings

        self.quotation_char = '"'  # flipped to "'" when inside an f-string with double quotes
        self.fstring_stack = []

    def __create_iid(self, node):
        location = self.get_metadata(PositionProvider, node)
        line = location.start.line
        column_start = location.start.column
        column_end = location.end.column
        iid = self.iids.new(self.file_path, line, column_start, column_end)
        return iid

    def __create_name_call(self, node, updated_node):
        callee_name = cst.Name(value="_n_")
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        name_arg = cst.Arg(cst.SimpleString(
            value=f"{self.quotation_char}{node.value}{self.quotation_char}"))
        lambada = cst.Lambda(params=cst.Parameters(
            params=[]), body=updated_node)
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

    def __get_callee_name_node(self, call_node):
        if isinstance(call_node.func, cst.Name):
            return call_node.func
        elif isinstance(call_node.func, cst.Attribute):
            return call_node.func.attr
        else: # everything else, e.g., cst.Subscript
            return call_node.func

    def __create_call_call(self, node, updated_node):
        callee_name = cst.Name(value="_c_")
        node_of_callee_name = self.__get_callee_name_node(node)
        iid = self.__create_iid(node_of_callee_name)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        fct_arg = cst.Arg(value=updated_node.func)
        all_args = [iid_arg, fct_arg] + \
            self.__ensure_generator_expr_have_parens(updated_node.args)
        call = cst.Call(func=callee_name, args=all_args)
        return call

    def __create_attribute_call(self, node, updated_node):
        callee_name = cst.Name(value="_a_")
        assert type(node.attr) == cst.Name, type(node.attr)
        iid = self.__create_iid(node.attr)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        value_arg = cst.Arg(updated_node.value)
        attr_arg = cst.Arg(cst.SimpleString(
            value=f"{self.quotation_char}{node.attr.value}{self.quotation_char}"))
        call = cst.Call(func=callee_name, args=[iid_arg, value_arg, attr_arg])
        return call
    
    def __create_line_call(self, node, updated_node):
        callee_name = cst.Name(value="_l_")
        iid = self.__create_iid(node)
        iid_arg = cst.Arg(value=cst.Integer(value=str(iid)))
        call = cst.Call(func=callee_name, args=[iid_arg])
        return call

    def __create_import(self, name):
        module_name = cst.Attribute(value=cst.Name(
            value="lexecutor"), attr=cst.Name(value="Runtime"))
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

        # assignments to a single value
        if (type(parent) == cst.AssignTarget or
                type(parent) == cst.AnnAssign or
                type(parent) == cst.AugAssign):
            return True

        # multi-assignments
        if type(parent) == cst.Element:
            grand_parent = self.get_metadata(ParentNodeProvider, parent)
            if type(grand_parent) == cst.Tuple:
                grand_grand_parent = self.get_metadata(
                    ParentNodeProvider, grand_parent)
                if (type(grand_grand_parent) == cst.AssignTarget or
                    type(grand_grand_parent) == cst.AnnAssign or
                        type(grand_grand_parent) == cst.AugAssign):
                    return True

        return False

    def __is_ignored_call(self, call_node):
        if type(call_node.func) == cst.Name:
            return call_node.func.value in self.ignored_calls
        else:
            return False

    def visit_SimpleStatementLine(self, node):
        # don't visit lines marked with special comment
        c = node.trailing_whitespace.comment
        if c is not None and c.value == "# don't instrument":
            return False
        return True

    def visit_Import(self, node):
        # don't instrument imports, as we'll wrap them in try-except
        return False

    def visit_ImportFrom(self, node):
        # don't instrument imports, as we'll wrap them in try-except
        return False

    def visit_Del(self, node):
        # don't instrument delete statements, as "del" on call on allowed
        return False

    def visit_FormattedString(self, node):
        if node.start == 'f"' or node.start == 'fr"' or node.start == 'rf"':
            self.quotation_char = "'"
            self.fstring_stack.append(node)
        elif node.start == "f'" or node.start == "fr'" or node.start == 'rf"':
            self.quotation_char = '"'
            self.fstring_stack.append(node)
        if len(self.fstring_stack) > 1:
            self.instrument = False
        return True

    def leave_FormattedString(self, node, updated_node):
        if self.fstring_stack and node == self.fstring_stack[-1]:
            # flip quotation character back
            if self.quotation_char == "'":
                self.quotation_char = '"'
            elif self.quotation_char == '"':
                self.quotation_char = "'"
            self.fstring_stack.pop()
            if len(self.fstring_stack) < 2:
                self.instrument = True
        return updated_node

    def leave_Call(self, node, updated_node):
        # rewrite Call nodes to intercept function calls
        if not self.__is_ignored_call(node) and not self.line_coverage_instrumentation:
            wrapped_call = self.__create_call_call(node, updated_node)
            return wrapped_call
        else:
            return updated_node

    def leave_Name(self, node, updated_node):
        if not self.instrument:
            return updated_node

        # rewrite Name nodes to intercept values they resolve to
        if node in self.used_names and node.value not in self.ignored_names and not self.line_coverage_instrumentation:
            wrapped_name = self.__create_name_call(node, updated_node)
            return wrapped_name
        else:
            return updated_node

    def leave_Attribute(self, node, updated_node):
        if not self.instrument:
            return updated_node

        if not self.__is_l_value(node) and not self.line_coverage_instrumentation:
            wrapped_attribute = self.__create_attribute_call(node, updated_node)
            return wrapped_attribute
        else:
            return updated_node

    def leave_SimpleStatementLine(self, node, updated_node):
        statement_call = self.__create_line_call(node, updated_node)
        if not self.instrument:
            return cst.FlattenSentinel([statement_call, cst.Expr(cst.Newline()), updated_node])

        # surround imports with try-except;
        # cannot do this in leave_Import because we need to replace the import's parent node
        if isinstance(node.body[0], cst.Import) or isinstance(node.body[0], cst.ImportFrom):
            # don't wrap __future__ imports
            if not (isinstance(node.body[0], cst.ImportFrom) and
                    node.body[0].module is not None and
                    node.body[0].module.value == "__future__"):
                # don't try-except-pass wrap imports that are already surrounded by try-except (as they should sometimes fail)
                skip = False
                parent = self.get_metadata(ParentNodeProvider, node)
                if isinstance(parent, cst.IndentedBlock):
                    grand_parent = self.get_metadata(
                        ParentNodeProvider, parent)
                    if isinstance(grand_parent, cst.Try):
                        skip = True
                if not skip:
                    wrapped_import = self.__wrap_import(
                        node.body[0], updated_node.body[0])
                    return wrapped_import
        return cst.FlattenSentinel([statement_call, cst.Expr(cst.Newline()), updated_node])
    
    def leave_For(self, node, updated_node):
        statement_call = self.__create_line_call(node, updated_node)
        return cst.FlattenSentinel([statement_call, cst.Expr(cst.Newline()), updated_node])
    
    def leave_While(self, node, updated_node):
        statement_call = self.__create_line_call(node, updated_node)
        return cst.FlattenSentinel([statement_call, cst.Expr(cst.Newline()), updated_node])
    
    def leave_FunctionDef(self, node, updated_node):
        statement_call = self.__create_line_call(node, updated_node)
        return cst.FlattenSentinel([statement_call, cst.Expr(cst.Newline()), updated_node])

    def leave_ClassDef(self, node, updated_node):
        statement_call = self.__create_line_call(node, updated_node)
        return cst.FlattenSentinel([statement_call, cst.Expr(cst.Newline()), updated_node])
    
    def leave_With(self, node, updated_node):
        statement_call = self.__create_line_call(node, updated_node)
        return cst.FlattenSentinel([statement_call, cst.Expr(cst.Newline()), updated_node])
    
    def leave_IndentedBlock(self, node, updated_node):
        new_body = []
        for i in range(len(updated_node.body)):
            stmt = updated_node.body[i]
            if (isinstance(stmt, cst.If)
               or isinstance(stmt, cst.Try)):
                statement_call = self.__create_line_call(node, updated_node)
                new_body.append(statement_call)
                new_body.append(cst.Expr(cst.Newline()))
            new_body.append(stmt)
            
        return updated_node.with_changes(body=new_body)
    
    def leave_Module(self, node, updated_node):
        if not self.instrument:
            return updated_node
        
        # check for "__future__" imports; they must remain at beginning of file
        target_idx = 0  # index to add our imports at
        new_body = []
        for i in range(len(updated_node.body)):
            stmt = updated_node.body[i]
            if (isinstance(stmt, cst.If)
               or isinstance(stmt, cst.Try)):
                statement_call = self.__create_line_call(node, updated_node)
                new_body.append(statement_call)
                new_body.append(cst.Expr(cst.Newline()))
            if (isinstance(stmt, cst.SimpleStatementLine)
               and isinstance(stmt.body[0], cst.ImportFrom)
               and stmt.body[0].module.value == "__future__"):
                target_idx = i + 1
            new_body.append(stmt)
            
        # add our imports
        import_n = self.__create_import("_n_")
        import_a = self.__create_import("_a_")
        import_c = self.__create_import("_c_")
        import_l = self.__create_import("_l_")

        new_body = (list(new_body[:target_idx])
                    + [import_n, import_a, import_c, import_l]
                    + list(new_body[target_idx:]))

        return updated_node.with_changes(body=new_body)
