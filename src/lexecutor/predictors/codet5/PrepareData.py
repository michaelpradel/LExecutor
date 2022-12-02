import argparse
import multiprocessing
import pandas as pd
import numpy as np
import torch as t
from ...Logging import logger
from ...Util import gather_files
from .CodeT5 import load_CodeT5
from ...Hyperparams import Hyperparams as params
from ...IIDs import IIDs
from .InputFactory import InputFactory


parser = argparse.ArgumentParser()
parser.add_argument(
    "--iids", help="JSON file with instruction IDs", required=True)
parser.add_argument(
    "--traces", help="Trace file or .txt file(s) with all trace file paths to use",
    nargs="+", required=True)
parser.add_argument(
    "--output_suffix", help="Suffix to append to output file names (if nothing given: train.pt, validate.pt)")


def read_traces(trace_files):
    logger.info("Loading trace files")
    name_df = pd.DataFrame(data=None)
    call_df = pd.DataFrame(data=None)
    attribute_df = pd.DataFrame(data=None)

    trace_files = gather_files(trace_files)
    for trace_file in trace_files:
        # TODO: really need try-except?
        try:
            current_name_df = pd.read_hdf(trace_file, key="name")
        except:
            logger.warning(f"Warning: Could not read trace file {trace_file}")
            current_name_df = pd.DataFrame(data=None)

        try:
            current_call_df = pd.read_hdf(trace_file, key="call")
        except:
            logger.warning(f"Warning: Could not read trace file {trace_file}")
            current_call_df = pd.DataFrame(data=None)

        try:
            current_attribute_df = pd.read_hdf(trace_file, key="attribute")
        except:
            logger.warning(f"Warning: Could not read trace file {trace_file}")
            current_attribute_df = pd.DataFrame(data=None)

        name_df = pd.concat([name_df, current_name_df])
        call_df = pd.concat([call_df, current_call_df])
        attribute_df = pd.concat(
            [attribute_df, current_attribute_df])

    # merge all entries into a single dataframe: iid, name, value
    name_df["kind"] = "name"
    call_df["kind"] = "call"
    attribute_df["kind"] = "attribute"
    entries = pd.concat([
        name_df[[0, 1, 2, "kind"]].rename(
            columns={0: "iid", 1: "name", 2: "value"}),
        call_df[[0, 1, 3, "kind"]].rename(
            columns={0: "iid", 1: "name", 3: "value"}),
        attribute_df[[0, 2, 3, "kind"]].rename(
            columns={0: "iid", 2: "name", 3: "value"})
    ])

    return entries


def dedup_trace_entries(entries):
    logger.info(f"Deduplicating {len(entries)} trace entries")
    if params.dedup == "name-value-iid":
        entries.drop_duplicates(inplace=True)
    elif params.dedup == "name-value":
        entries.drop_duplicates(subset=["name", "value"], inplace=True)
    else:
        raise ValueError(f"Unknown dedup mode: {params.dedup}")

    # TODO the following handles some bug in trace gathering, which should be fixed there
    entries.drop(entries[entries.name.astype(str).str.startswith(
        "MarkDecorator")].index, inplace=True)

    logger.info(f"After deduplicating: {len(entries)} trace entries")


def split_and_shuffle(entries, iids):
    # split
    if params.split == "project":
        # TODO implement
        pass
    elif params.split == "file":
        # TODO implement
        pass
    elif params.split == "mixed":
        mask = np.random.randn(len(entries)) < 0.8
        train_entries = entries[mask]
        validate_entries = entries[~mask]
    else:
        raise ValueError(f"Unknown split mode: {params.split}")

    # shuffle
    train_entries = train_entries.sample(frac=1).reset_index()
    validate_entries = validate_entries.sample(frac=1).reset_index()

    logger.info(
        f"{len(train_entries)} training entries, {len(validate_entries)} validation entries")
    return train_entries, validate_entries


def gather_context_and_vectorize(entries, iids, tokenizer):
    factory = InputFactory(iids, tokenizer)

    all_vectorized = t.empty(
        [len(entries), 512+params.max_output_length], dtype=t.long)
    for index, entry in entries.iterrows():
        input_ids, label_ids = factory.entry_to_inputs(entry)
        all_vectorized[index] = t.cat([input_ids, label_ids])

        if index % 10000 == 0:
            logger.info(f"Vectorized {index}/{len(entries)} entries")

    logger.info(f"Created tensor of shape {all_vectorized.shape}")
    return all_vectorized


def store_tensors(train_tensors, validate_tensors, output_suffix):
    if output_suffix is None:
        train_path = f"train{output_suffix if output_suffix is not None else ''}.pt"
        validate_path = f"validate{output_suffix if output_suffix is not None else ''}.pt"
    t.save(train_tensors, train_path)
    t.save(validate_tensors, validate_path)
    logger.info(f"Stored tensors to {train_path} and {validate_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer, model = load_CodeT5()

    iids = IIDs(args.iids)
    entries = read_traces(args.traces)
    dedup_trace_entries(entries)
    train_entries, validate_entries = split_and_shuffle(entries, iids)

    train_tensors = gather_context_and_vectorize(
        train_entries, iids, tokenizer)
    validate_tensors = gather_context_and_vectorize(
        validate_entries, iids, tokenizer)

    store_tensors(train_tensors, validate_tensors, args.output_suffix)
