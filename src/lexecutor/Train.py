import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from gensim.models.fasttext import FastText
from .Hyperparams import Hyperparams as p
from .TensorFactory import TensorFactory
from .Training import Training
from .Validation import Validation
from .Model import ValuePredictionModel
from .Util import device

parser = argparse.ArgumentParser()
parser.add_argument(
    "--traces", help="Trace files", nargs="+", required=False)
parser.add_argument(
    "--train_tensors", help="Directory with .npz files with pre-computed tensors to use for training", required=False)
parser.add_argument(
    "--validate_tensors", help="Directory with .npz files with pre-computed tensors to use for validation", required=False)
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
    
    tensor_factory = TensorFactory()

    if args.traces is not None:
        embedding = load_FastText(args.embedding)
        tensor_factory.traces_to_tensors(args.traces, embedding, "data/tensors")
    elif args.train_tensors is not None:
        model = ValuePredictionModel().to(device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters())

        train_dataset = tensor_factory.tensors_as_dataset(args.train_tensors)
        train_loader = DataLoader(train_dataset, batch_size=p.batch_size)
        training = Training(model, criterion, optimizer,
                            train_loader, p.batch_size, p.epochs)

        validate_dataset = tensor_factory.tensors_as_dataset(args.validate_tensors)
        validate_loader = DataLoader(validate_dataset, batch_size=p.batch_size)
        validation = Validation(
            model, criterion, validate_loader, p.batch_size)

        training.run(validation=validation)
