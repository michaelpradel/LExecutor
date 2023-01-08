class Hyperparams(object):
    iids_file = "iids.json"
    verbose = False
    
    # data deduplication
    # dedup = "name-value"
    dedup = "name-value-iid"

    # data splitting
    # split = "project"
    # split = "file"
    split = "mixed"

    perc_train = 0.95

    # CodeT5 model
    max_output_length = 8

    # feedforward model
    token_emb_len = 100
    value_emb_len = 20
    max_call_args = 3
    joined_layer_len = 200
    intermediate_layer_len = 200

    # training
    epochs = 10
    batch_size = 50

    
