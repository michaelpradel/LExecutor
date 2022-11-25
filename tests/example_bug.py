if (not has_min_size(all_data)):
    raise RuntimeError("not enough data")

train_len = 0.8 * len(all_data)

logger.info(f"Extracting training data with config {config_str}")

train_data = all_data[0:train_len]
print(train_data)