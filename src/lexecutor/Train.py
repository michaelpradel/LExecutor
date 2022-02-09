import argparse
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from gensim.models.fasttext import FastText
from .TraceReader import read_trace
from .Hyperparams import Hyperparams as p
from .TensorFactory import TensorFactory
from .Training import Training
from .Validation import Validation
from .Model import ValuePredictionModel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--traces", help="Trace files", nargs="+", required=False)
parser.add_argument(
    "--tensors", help=".npz file with pre-computed tensors", required=False)
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
    tensor_factory = TensorFactory(embedding)
    if args.traces is not None:
        tensor_factory.traces_to_tensors(args.traces, "data/tensors")
    elif args.tensors is not None:
        
        model = ValuePredictionModel().to(device)
        criterion = BCELoss()
        optimizer = Adam(model.parameters())

        # dataset = ToyDataset(10000, 20, 10) # for testing only

        train_loader = create_dataloader(
            json_train_dataset, embedding, embedding_size)
        training = Training(model, criterion, optimizer,
                            train_loader, batch_size, epochs)

        validate_loader = create_dataloader(
            json_validate_dataset, embedding, embedding_size)
        validation = Validation(model, criterion, validate_loader, batch_size)
        
        training.run(args.store_model, validation)
