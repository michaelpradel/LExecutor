import argparse
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from gensim.models.fasttext import FastText
from .TraceReader import read_trace
from .Hyperparams import Hyperparams as p
from .TensorFactory import TensorFactory


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
        tensor_factory.traces_to_tensors(args.traces, "tensors.npz")
    # TODO: use .npz file, if given
