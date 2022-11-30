import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from gensim.models.fasttext import FastText
from ...Hyperparams import Hyperparams as p
from .TensorFactory import TensorFactory
from .Training import Training
from .Validation import Validation
from .Model import ValuePredictionModel
from ...Util import device
from .TraceToTensorDataset import TraceToTensorDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_trace", help="Trace file or .txt file with all trace file paths to use for training", nargs="+", required=True)
parser.add_argument(
    "--validate_trace", help="Trace file or .txt file with all trace file paths to use for validation", nargs="+", required=True)
parser.add_argument(
    "--embedding", help="Pre-trained FastText token embedding", required=True)


def load_CodeBERT():
    print("Loading pre-trained CodeBERT token embedding")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    return tokenizer, model


def load_FastText(file_path):
    print("Loading pre-trained FastText token embedding")
    embedding = FastText.load(file_path)
    embedding_size = len(embedding.wv["test"])
    if embedding_size != p.token_emb_len:
        raise Exception(
            "FastText embedding size does not match Hyperparams.token_emb_len")
    return embedding


def name_to_vectors(name, tokenizer, model):
    tokens = tokenizer.tokenize(name)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
    return embeddings[0]


if __name__ == "__main__":
    args = parser.parse_args()

    embedding = load_FastText(args.embedding)
    print(embedding)

    model = ValuePredictionModel().to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    tensor_factory = TensorFactory(embedding)
    train_dataset = TraceToTensorDataset(args.train_trace, tensor_factory)
    train_loader = DataLoader(
        train_dataset, batch_size=p.batch_size, drop_last=True)
    print(f"Training on   {len(train_dataset)} examples")
    training = Training(model, criterion, optimizer,
                        train_loader, p.batch_size, p.epochs)

    validate_dataset = TraceToTensorDataset(
        args.validate_trace, tensor_factory)
    validate_loader = DataLoader(
        validate_dataset, batch_size=p.batch_size, drop_last=True)
    print(f"Validating on {len(validate_dataset)} examples")
    validation = Validation(
        model, criterion, validate_loader, p.batch_size)

    training.run(validation=validation,
                 store_model_path="data/models/latest")

    tensor_factory.save_value_map()
