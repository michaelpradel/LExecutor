import torch
from torch import nn
import pytorch_lightning as pl
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

# TODO RAD
if __name__ == '__main__':
    

class ValuePredictionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
