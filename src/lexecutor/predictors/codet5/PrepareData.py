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
    df = pd.DataFrame(data=None)
    trace_files = gather_files(trace_files)
    for trace_file in trace_files:
        current_df = pd.read_hdf(trace_file, key="entries")
        df = pd.concat([df, current_df])
    return df


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


def clean_entries(entries):
    before_len = len(entries)
    # remove entries with invalid names (e.g. "functools.partial(<bound")
    entries.drop(entries[entries.name.astype(
        str).str.find("(") != -1].index, inplace=True)
    logger.info(
        f"Data cleaning removes {before_len - len(entries)} of {before_len} entries")


def split_and_shuffle(entries, iids):
    # split
    if params.split == "project":
        # TODO implement
        pass
    elif params.split == "file":
        # TODO implement
        pass
    elif params.split == "mixed":
        logger.info(
            f"Mixed split with {params.perc_train} of entries for training")
        mask = np.random.random(len(entries)) < params.perc_train
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
    clean_entries(entries)
    train_entries, validate_entries = split_and_shuffle(entries, iids)

    train_tensors = gather_context_and_vectorize(
        train_entries, iids, tokenizer)
    validate_tensors = gather_context_and_vectorize(
        validate_entries, iids, tokenizer)

    store_tensors(train_tensors, validate_tensors, args.output_suffix)
