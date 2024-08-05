from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier
from utils import parse_config, preprocess_data
from dataloader import LibriDataset


def save_pred(pred: tuple[torch.Tensor, torch.Tensor], csv_path: Path) -> None:
    with open(csv_path, mode="w") as f:
        f.write("Id,Class\n")
        for i, y in enumerate(pred):
            f.write(f"{i},{y}\n")


def test_model(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, features in enumerate(tqdm(test_loader)):
            features = features.to(device)

            outputs = model(features)

            _, test_pred = torch.max(outputs, 1)
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)
    save_pred(pred, "./pred.csv")


if __name__ == "__main__":
    config = parse_config()
    input_dim = 39 * config["concat_nframes"]
    test_X = preprocess_data(
        split="test",
        feat_dir="./libriphone/feat",
        phone_path="./libriphone",
        concat_num=config["concat_nframes"],
    )
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)
    model = Classifier(
        input_dim=input_dim,
        hidden_layers=config["hidden_layers"],
        hidden_dim=config["hidden_dim"],
    )
    model.load_state_dict(torch.load(config["model_path"]))
    test_model(model, test_loader)
