from pathlib import Path
import random
import numpy as np
import torch
import os
import yaml
import logging


def load_feature(path: Path) -> torch.Tensor:
    return torch.load(path)


def concat_features(x: torch.Tensor, concat_num: int) -> torch.Tensor:
    if concat_num % 2 == 0:
        raise ValueError("concat_num must be odd")
    if concat_num == 1:
        return x
    seq_len, feature_dim = x.shape[0], x.shape[1]
    x = x.repeat(1, concat_num)
    x = x.view(seq_len, concat_num, feature_dim).permute(1, 0, 2)
    mid: int = concat_num // 2
    for i in range(1, mid + 1):
        x[mid + i] = rotate_left(x[mid + i], i)
        x[mid - i] = rotate_right(x[mid - i], i)
    return x.permute(1, 0, 2).view(seq_len, concat_num * feature_dim)


def rotate_right(x: torch.Tensor, idx: int) -> torch.Tensor:
    left: torch.Tensor = x[0].repeat(idx, 1)
    right: torch.Tensor = x[:-idx]
    return torch.cat((left, right), dim=0)


def rotate_left(x: torch.Tensor, idx: int) -> torch.Tensor:
    left: torch.Tensor = x[idx:]
    right: torch.Tensor = x[-1].repeat(idx, 1)
    return torch.cat((left, right), dim=0)


def parse_config() -> dict:
    config_path = "./config/config.yaml"
    with open(config_path) as f:
        config = yaml.load(f, yaml.Loader)
    return config


def set_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s- [%(levelname)s]: %(message)s")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def preprocess_data(
    split: str,
    feat_dir: Path,
    phone_path: Path,
    concat_num: int,
    train_ratio: int = 0.8,
    train_val_seed: int = 1337,
):
    assert (
        split == "train" or split == "val" or split == "test"
    ), "Invalid split argument."
    class_num: int = 41
    mode = "train" if (split == "train" or split == "val") else "test"

    label_dict = {}
    if mode == "train":
        with open(os.path.join(phone_path, "train_labels.txt")) as f:
            files_and_labels = f.readlines()
        for line in files_and_labels:
            line = line.strip().split(" ")
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if mode == "train":
        with open(os.path.join(phone_path, "train_split.txt")) as f:
            feat_filename: list[str] = f.readlines()
        random.seed(train_val_seed)
        random.shuffle(feat_filename)
        cut_point = int(len(feat_filename) * train_ratio)
        feat_filename = (
            feat_filename[:cut_point] if split == "train" else feat_filename[cut_point:]
        )
    else:
        with open(os.path.join(phone_path, "test_split.txt")) as f:
            feat_filename: list[str] = f.readlines()

    feat_filename = [line.strip() for line in feat_filename]
    print(
        f"[Dataset] - # phone classes: {class_num}, number of utterances for {split}: {len(feat_filename)}"
    )

    max_len = int(3e6)
    X: torch.Tensor = torch.empty(max_len, 39 * concat_num)
    if mode == "train":
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for _, fname in enumerate(feat_filename):
        feat: torch.Tensor = load_feature(os.path.join(feat_dir, mode, f"{fname}.pt"))
        seq_len: int = len(feat)
        feat = concat_features(feat, concat_num)
        if mode == "train":
            label: torch.Tensor = torch.LongTensor(label_dict[fname])
            y[idx : idx + seq_len] = label

        X[idx : idx + seq_len] = feat
        idx += seq_len

    X = X[:idx]
    if mode == "train":
        y = y[:idx]

    print(f"[INFO] {split} set: {X.shape}")
    if mode == "train":
        print(f"label shape: {y.shape}")
        return X, y
    else:
        return X


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    config = parse_config()
    assert isinstance(config, dict)
    print(config)
