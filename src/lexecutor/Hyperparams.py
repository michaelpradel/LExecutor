class Hyperparams(object):
    # data deduplication
    dedup = "name-value"
    # dedup = "name-value-iid"

    # data splitting
    # split = "project"
    # split = "file"
    split = "mixed"

    # input encoding
    token_emb_len = 100
    value_emb_len = 20
    max_call_args = 3

    # neural model
    joined_layer_len = 200
    intermediate_layer_len = 200

    # training
    epochs = 2
    batch_size = 50
