# from https://github.com/mininet/mininet/blob/master/mininet/term.py

def fut():
    """Create terminals.
       nodes: list of Node objects
       title: base title for each
       returns: list of created tunnel/terminal processes"""
    terms = []
    for node in nodes:
        terms += makeTerm( node, title, term )
    return terms


if __name__ == "__main__":
    fut()
