import argparse
import json
from collections import Counter
from .codet5.PrepareData import read_traces, clean_entries

parser = argparse.ArgumentParser()
parser.add_argument(
    "--traces", help="Trace file or .txt file(s) with all trace file paths to use",
    nargs="+", required=True)

def get_values_frequencies(trace_files):
    name_to_values = {}
    call_to_values = {}
    attribute_to_values = {}
    
    entries = read_traces(trace_files)
    clean_entries(entries)
    for index, entry in entries.iterrows():
        key = entry["name"]
        if entry["kind"] == "name":
            name_to_values.setdefault(key, Counter())[
                entry.value] += 1
        elif entry["kind"] == "call":
            call_to_values.setdefault(key, Counter())[
                entry.value] += 1
        elif entry["kind"] == "attribute":
            attribute_to_values.setdefault(key, Counter())[
                entry.value] += 1
            
    return {
        "name_to_values": name_to_values,
        "call_to_values": call_to_values,
        "attribute_to_values": attribute_to_values
    }
            
def store_values_frequencies(values_frequencies):
    with open("values_frequencies.json", "w") as outfile:
        json.dump(values_frequencies, outfile)
    
if __name__ == "__main__":
    args = parser.parse_args()
    values_frequencies = get_values_frequencies(args.traces)
    store_values_frequencies(values_frequencies)