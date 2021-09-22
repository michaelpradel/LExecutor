def original():
    my_array = [1, 2, 3]
    x = a
    y = bar(x)
    z = my_array[my_index]


# _t_(lambda: ORIG)

def instrumented():
    my_array = [1, 2, 3]
    x = _t_(lambda: a, "a")
    print(x)
    print()
    
    y = _t_(lambda: _t_(lambda: bar, "bar")(_t_(lambda: x, "x")), "bar(x)")
    print(y)
    print()

    z = _t_(lambda: _t_(lambda: my_array, "my_array")[_t_(lambda: my_index, "my_index")], "lookup")
    print(z)


def _t_(read_value, info=None):
    try:
        v = read_value()
        print(f"{info}: Got actual value {v}")
        return v
    except Exception as e:
        v = 1
        print(f"{info}: Will return default value {v} because {e}")
        return v


if __name__ == "__main__":
    # original()
    instrumented()
