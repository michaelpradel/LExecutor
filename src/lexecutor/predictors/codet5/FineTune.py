import os
import argparse
import random
import torch as t
import csv
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
from .CodeT5 import load_CodeT5
from ...Hyperparams import Hyperparams as params
from ...Util import device
from ...Logging import logger


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_tensors", help=".pt files for training", default="train.pt")
parser.add_argument(
    "--validate_tensors", help=".pt files for validation", default="validate.pt")
parser.add_argument(
    "--output_dir", help="directory to store models", required=True)
parser.add_argument(
    "--save_last_checkpoints", help="Save checkpoint after every 100 batches", action="store_true")


print_examples = True


def evaluate(validate_tensors_path, model, tokenizer):
    validate_dataset = TensorDataset(t.load(validate_tensors_path))
    validate_loader = DataLoader(
        validate_dataset, batch_size=params.batch_size, drop_last=True)

    logger.info("Starting evaluation")
    logger.info("  Num examples = {}".format(len(validate_dataset)))
    logger.info("  Num batches = {}".format(len(validate_loader)))
    logger.info("  Batch size = {}".format(params.batch_size))

    accuracies = []

    with t.no_grad():
        model.eval()

        for batch_idx, batch in enumerate(validate_loader):
            batch = t.cat(batch)
            input_ids = batch[:, 0:512]
            input_ids = input_ids.to(device)
            label_ids = batch[:, 512:518]
            label_ids = label_ids.to(device)

            labels = tokenizer.batch_decode(
                label_ids, skip_special_tokens=True)

            generated_ids = model.generate(
                input_ids, max_length=params.max_output_length)
            predictions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

            corrects = [1 for i in range(
                len(labels)) if labels[i] == predictions[i]]
            accuracies_batch = float(len(corrects)) / len(labels)
            accuracies.append(accuracies_batch)

            # for debugging
            if print_examples:
                for label_idx, label in enumerate(labels):
                    if random.uniform(0, 100) < 0.1:
                        prediction = predictions[label_idx]
                        logger.info(f"Label: {label}, Prediction: {prediction}")

    val_accuracy = np.array(accuracies).mean().item()
    logger.info(
        f"val_accuracy = {round(val_accuracy, 4)}")

    logger.info('Terminating evaluation')


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer, model = load_CodeT5()

    train_dataset = TensorDataset(t.load(args.train_tensors))
    train_loader = DataLoader(
        train_dataset, batch_size=params.batch_size, drop_last=True)

    optim = AdamW(model.parameters(), lr=1e-5)

    logger.info(f"Starting training on {device}")
    logger.info("  Num examples = {}".format(len(train_dataset)))
    logger.info("  Batch size = {}".format(params.batch_size))
    logger.info("  Batch num = {}".format(
        len(train_dataset) / params.batch_size))
    logger.info("  Num epoch = {}".format(params.epochs))

    # save last checkpoint
    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
    if not os.path.exists(last_output_dir):
        os.makedirs(last_output_dir)

    for epoch in range(params.epochs):
        logger.info(f"Epoch {epoch}")

        for batch_idx, batch in enumerate(train_loader):
            batch = t.cat(batch)
            input_ids = batch[:, 0:512]
            input_ids = input_ids.to(device)
            labels = batch[:, 512:518]
            labels = labels.to(device)

            model.train()
            optim.zero_grad()

            outputs = model(input_ids, labels=labels)

            loss = outputs.loss
            loss.backward()
            optim.step()

            logger.info(
                f"  Training loss of batch {batch_idx}: {round(loss.item(), 4)}")

            if (batch_idx+1) % 100 == 0 and args.save_last_checkpoints:
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                output_model_file = os.path.join(
                    last_output_dir, "pytorch_model.bin")
                t.save(model_to_save.state_dict(), output_model_file)
                logger.info("Saved the last model into %s", output_model_file)
                evaluate(args.validate_tensors, model, tokenizer)

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
            df = pd.concat([df, df_new_data])
            df.to_csv(f'./training_loss.csv', index=False)

        evaluate(args.validate_tensors, model, tokenizer)

    logger.info('Terminating training')
