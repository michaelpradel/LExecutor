import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from .TraceReader import read_trace


def load_CodeBERT():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    return tokenizer, model


def name_to_vectors(name, tokenizer, model):
    tokens = tokenizer.tokenize(name)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
    return embeddings[0]


def prepare_data(entries, tokenizer, model):
    xs = []
    ys = []


if __name__ == "__main__":
    tokenizer, model = load_CodeBERT()
    entries = read_trace("../data/trace_sample.txt")
    prepare_data(entries, tokenizer, model)
