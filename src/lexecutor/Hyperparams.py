class Hyperparams(object):
    # data deduplication
    # dedup = "name-value"
    dedup = "name-value-iid"

    # data splitting
    # split = "project"
    # split = "file"
    split = "mixed"

    # CodeT5 model
    max_output_length = 8

    # feedforward model
    token_emb_len = 100
    value_emb_len = 20
    max_call_args = 3
    joined_layer_len = 200
    intermediate_layer_len = 200

    # training
    epochs = 2
    batch_size = 50
