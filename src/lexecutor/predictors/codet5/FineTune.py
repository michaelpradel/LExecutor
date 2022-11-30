import os
import math
import argparse
import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from .InputFactory import InputFactory
from .MaskedValueDataset import MaskedValueDataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AdamW
from ...Hyperparams import Hyperparams as p
from datetime import datetime

from ...Util import device

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_trace", help="Trace file or .txt file with all trace file paths to use for training", nargs="+", required=True)
parser.add_argument(
    "--validate_trace", help="Trace file or .txt file with all trace file paths to use for validation", nargs="+", required=True)
parser.add_argument(
    "--iids", help="JSON file with instruction IDs", required=True)
parser.add_argument(
    "--output_dir", help="directory to store models", required=True)
parser.add_argument(
    "--save_last_checkpoints", help="True if the model should be saved after every batch and False otherwise", required=True)


def load_CodeT5():
    print("Loading pre-trained codet5-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    model = T5ForConditionalGeneration.from_pretrained(
        'Salesforce/codet5-small')
    model.to(device)
    return tokenizer, model


def evaluate(args, model, tokenizer, input_factory):
    validate_dataset = MaskedValueDataset(args.validate_trace, input_factory)
    validate_loader = DataLoader(
        validate_dataset, batch_size=p.batch_size, drop_last=True)

    # Eval!
    print("Starting evaluation: {}".format(
        datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
    print("  Num examples = {}".format(len(validate_dataset)))
    print("  Num batches = {}".format(len(validate_loader)))
    print("  Batch size = {}".format(p.batch_size))

    accuracies = []

    with torch.no_grad():
        model.eval()

        for batch_idx, batch in enumerate(validate_loader):
            input_ids = batch['input_ids'].to(device)
            labels_ids = batch['labels'].to(device)
            labels = [tokenizer.decode(ids, skip_special_tokens=True)
                      for ids in labels_ids]

            generated_ids = model.generate(input_ids, max_length=7)
            predictions = [tokenizer.decode(
                ids, skip_special_tokens=True) for ids in generated_ids]

            corrects = [1 for i in range(
                len(labels)) if labels[i] == predictions[i]]

            accuracies_batch = float(len(corrects)) / len(labels)

            accuracies.append(accuracies_batch)

    val_accuracy = np.array(accuracies).mean().item()
    print(
        f"val_accuracy = {round(val_accuracy, 4)}")

    print('Terminating evaluation: {}'.format(
        datetime.now().strftime("%m/%d/%Y %H:%M:%S")))


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer, model = load_CodeT5()

    input_factory = InputFactory(tokenizer, args.iids)

    train_dataset = MaskedValueDataset(args.train_trace, input_factory)
    train_loader = DataLoader(
        train_dataset, batch_size=p.batch_size, drop_last=True)

    optim = AdamW(model.parameters(), lr=1e-5)

    print("Starting training: {}".format(
        datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Batch size = {}".format(p.batch_size))
    print("  Batch num = {}".format(len(train_dataset) / p.batch_size))
    print("  Num epoch = {}".format(p.epochs))

    for epoch in range(p.epochs):
        print(f"Epoch {epoch}")

        for batch_idx, batch in enumerate(train_loader):
            model.train()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optim.zero_grad()

            outputs = model(input_ids, labels=labels)

            loss = outputs.loss
            loss.backward()
            optim.step()

            print(
                f"  Training loss of batch {batch_idx}: {round(loss.item(), 4)}")

           # evaluate(args, model)

            # save last checkpoint
            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)

            if args.save_last_checkpoints:
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                output_model_file = os.path.join(
                    last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                print("Saved the last model into %s", output_model_file)

            # save last training loss
            if not os.path.isfile(f'./training_loss.csv'):
                columns = ['batch', 'loss']

                with open(f'./training_loss.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(columns)

            df = pd.read_csv(f'./training_loss.csv')
            df_new_data = pd.DataFrame({
                'batch': [batch_idx],
                'loss': [round(loss.item(), 4)],
                'epoch': [epoch]
            })
            df = df.append(df_new_data)
            df.to_csv(f'./training_loss.csv', index=False)

    print('Terminating training: {}'.format(
        datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
