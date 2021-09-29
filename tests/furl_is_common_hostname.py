# from https://github.com/gruns/furl/blob/master/furl/furl.py


def function_under_test():
    toks = hostname.split('.')
    if toks[-1] == '':  # Trailing '.' in a fully qualified domain name.
        toks.pop()

    for tok in toks:
        if is_valid_host.regex.search(tok) is not None:
            return False

    return '' not in toks  # Adjacent periods aren't allowed.


if __name__ == '__main__':
    function_under_test()
