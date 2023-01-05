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
    all_inputs = []
    all_labels = []
    all_predictions = []

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
            all_inputs.extend(tokenizer.batch_decode(
                input_ids, skip_special_tokens=False))
            all_labels.extend(labels)
            all_predictions.extend(predictions)
            if print_examples:
                for label_idx, label in enumerate(labels):
                    if random.uniform(0, 100) < 0.1:
                        prediction = predictions[label_idx]
                        logger.info(
                            f"Label: {label}, Prediction: {prediction}")

    val_accuracy = np.array(accuracies).mean().item()
    logger.info(
        f"val_accuracy = {round(val_accuracy, 4)}")

    # for debugging
    logger.info("Storing examples in human-readable format")
    examples_df = pd.DataFrame(
        {"input": all_inputs, "label": all_labels, "prediction": all_predictions})
    examples_df.to_pickle("./eval_examples.pkl")

    logger.info('Done with evaluation')
    return val_accuracy


def save_model(model, output_dir, epoch):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        output_dir, f"pytorch_model_epoch{epoch}.bin")
    t.save(model_to_save.state_dict(), output_model_file)
    logger.info("Saved the last model into %s", output_model_file)


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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df_training_loss = pd.DataFrame(columns=['batch', 'loss', 'epoch'])
    df_validation_acc = pd.DataFrame(columns=['epoch', 'val_accuracy'])

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

            # save training losses to file
            df_training_loss = pd.concat([df_training_loss, pd.DataFrame({
                'batch': [batch_idx],
                'loss': [round(loss.item(), 4)],
                'epoch': [epoch]
            })])
            df_training_loss.to_csv('./training_loss.csv', index=False)

        val_accuracy = evaluate(args.validate_tensors, model, tokenizer)

        # save validation accuracies to file
        df_validation_acc = pd.concat([df_validation_acc, pd.DataFrame({
            "epoch": [epoch],
            "val_accuracy": [val_accuracy]
        })])
        df_validation_acc.to_csv('./validation_acc.csv', index=False)

        save_model(model, args.output_dir, epoch)

    logger.info('Terminating training')
