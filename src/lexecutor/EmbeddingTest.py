from .Train import load_FastText

if __name__ == "__main__":
    embedding = load_FastText("data/embeddings/default/embedding")
    print(embedding.wv["test"])
    print(embedding.wv["apple"])
    print(embedding.wv["seq"])
    print(embedding.wv["list"])
    print(embedding.wv[";"])
