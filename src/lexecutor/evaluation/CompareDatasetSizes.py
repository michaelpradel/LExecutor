import os
from os.path import join, exists
import argparse
import csv
import torch as t
from ..Hyperparams import Hyperparams as params
from subprocess import run

description = """
Trains LExecutor on increasingly large datasets.
Usage:
 1) Prepare the traces by transforming them into tensors:
    CompareDatasetSizes --prepare --tensors <.pt files> --out_dir <tensor folder>
 2) Train LExecutor on increasingly large datasets:
    CompareDatasetSizes --train --in_dir <tensor folder> [--size <indices>]
 3) Produce raw results (to be used for plotting, etc.):
    CompareDatasetSizes --stats --in_dir <tensor folder>
"""
parser = argparse.ArgumentParser(description=description)
parser.add_argument("--prepare", action="store_true")
parser.add_argument(
    "--tensors", help=".pt files with data to use for training and validation (pass when using --prepare)", nargs="+")
parser.add_argument(
    "--out_dir", help="output folder for re-arranged training and validation data (pass when using --prepare)")
parser.add_argument("--train", action="store_true")
parser.add_argument(
    "--in_dir", help="folder with training and validation produced with --prepare (pass when using --train)")
parser.add_argument(
    "--size", help="fix the index of the run (optional; one index or comma-separated indices; pass when using --train)")
parser.add_argument(
    "--stats", help="summarize results of the experiment", action="store_true")


def load_data(tensor_files):
    print(f"Reading {len(tensor_files)} tensor files")
    tensors = []
    for tensor_file in tensor_files:
        tensors.append(t.load(tensor_file))
    combined = t.cat(tensors, 0)
    print(f"Combined tensors into one dataset of size {len(combined)}")    
    return combined


def rearrange_data(all_data):
    # compute indices for training (w/ increasing datasets) and validation
    shuffled_indices = t.randperm(len(all_data))
    train_indices = shuffled_indices[:int(params.perc_train * len(all_data))]
    validate_indices = shuffled_indices[int(params.perc_train * len(all_data)):]
    one_portion_length = int(len(train_indices) / 10)
    increasing_train_data_subset_indices = []
    for idx in range(0, 10):
        subset = train_indices[:one_portion_length * (idx + 1)]
        increasing_train_data_subset_indices.append(subset)
    
    # create tensors based on indices
    train_data_subsets = []
    for subset_idx, subset in enumerate(increasing_train_data_subset_indices):
        next_subset = all_data.index_select(0, subset)
        train_data_subsets.append(next_subset)
        print(f"Subset {subset_idx} has {len(next_subset)} data points")
    validate_data = all_data.index_select(0, validate_indices)

    return train_data_subsets, validate_data


def store_data(increasing_train_data_subsets, validate_data, out_dir):
    for idx, train_data_subset in enumerate(increasing_train_data_subsets):
        out_file = f"{out_dir}/train{idx}.pt"
        print(f"Storing training data subset with {len(train_data_subset)} data points into {out_file}")
        t.save(train_data_subset, out_file)
    t.save(validate_data, f"{out_dir}/validate.pt")
    print(f"Stored training and validation data into {out_dir}")


def run_training_with_size(in_dir, idx):
    print(f"Training on subset {idx}")
    stats_dir = f"{in_dir}/stats{idx}"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    run(["python", "-m", "lexecutor.predictors.codet5.FineTune",
         "--train_tensors", f"{in_dir}/train{idx}.pt",
         "--validate_tensors", f"{in_dir}/validate.pt",
         "--output_dir", f"{in_dir}/model{idx}",
         "--stats_dir", stats_dir])


def print_stats(in_dir):
    total_value_use_pairs = 0
    
    validation_file = f"{in_dir}/validate.pt"
    nb_validation_pairs = len(load_data([validation_file]))
    print(f"Validation pairs: {nb_validation_pairs}")
    total_value_use_pairs = nb_validation_pairs
    
    train_sizes = []
    accuracies = []
    for i in range(10):
        training_file = f"{in_dir}/train{i}.pt"
        nb_training_pairs = len(load_data([training_file]))
        stats_dir = f"{in_dir}/stats{i}"
        accuracy_file = join(stats_dir, "validation_acc.csv")
        if exists(accuracy_file):  # otherwise, experiment hasn't finished yet
            with open(accuracy_file, "r") as fp:
                accuracy_reader = csv.reader(fp)
                rows = list(accuracy_reader)
                if len(rows) == 6:
                    # experiment has finished for this dataset size
                    acc = rows[5][1]
                    train_sizes.append(str(nb_training_pairs))
                    accuracies.append(acc)
                    total_value_use_pairs += nb_training_pairs

    print(f"train_sizes = [{', '.join(train_sizes)}]")
    print(f"accuracies = [{', '.join(accuracies)}]")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.prepare:
        all_data = load_data(args.tensors)
        increasing_train_data_subsets, validate_data = rearrange_data(all_data)
        store_data(increasing_train_data_subsets, validate_data, args.out_dir)
    elif args.train:
        if args.size:
            if "," in args.size:
                sizes = [int(s) for s in args.size.split(",")]
            else:
                sizes = [int(args.size)]
        else:
            sizes = list(range(0, 10))
        
        for idx in sizes:
            run_training_with_size(args.in_dir, idx)
    elif args.stats:
        print_stats(args.in_dir)

