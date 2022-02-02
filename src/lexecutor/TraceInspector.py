import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--trace", help="Trace file")

if __name__ == "__main__":
    args = parser.parse_args()
    names = set()
    bin_op_combinations = set()
    with open(args.trace) as fp:
        for line in fp.readlines():
            segments = line.rstrip().split(" ")
            kind = segments[1]
            if kind == "name":
                names.add(segments[2])
            elif kind == "call":
                names.add(segments[2])
            elif kind == "attribute":
                names.add(segments[3])
            elif kind == "binary_operation":
                bin_op_combinations.add(f"{segments[2]} {segments[3]} {segments[4]}")
    print(f"{len(names)} unique names and {len(bin_op_combinations)} unique combinations of binary operation")
