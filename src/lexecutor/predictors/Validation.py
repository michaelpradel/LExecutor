# Copied from iCoDL

import torch as t


class Validation():
    def __init__(self, model, criterion, data_loader, batch_size):
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.batch_size = batch_size

    def run(self):
        # TODO: get nb from dataset (using non-iterable dataset?)
        accuracies = t.empty(100000)
        losses = t.empty(100000)
        ctr = 0
        with t.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(self.data_loader):
                xs = batch[0:-1]
                ys = batch[-1]

                ctr += ys.shape[0]

                ys_pred = self.model(xs)
                losses_batch = self.criterion(ys_pred, ys)
                ys_pred_winners = ys_pred.argmax(dim=1)
                ys_winners = ys.argmax(dim=1)

                ys_corrects = (ys_pred_winners == ys_winners)
                accuracies_batch = ys_corrects.sum().float() / ys.size(0)

                accuracies[batch_idx *
                           self.batch_size: (batch_idx + 1) * self.batch_size] = accuracies_batch
                losses[batch_idx *
                       self.batch_size: (batch_idx + 1) * self.batch_size] = losses_batch

        val_accuracy = accuracies.mean().item()
        val_loss = losses.mean().item()
        print(
            f"val_loss = {round(val_loss, 4)}, val_accuracy = {round(val_accuracy, 4)}")
        print(f"CTR = {ctr}")
