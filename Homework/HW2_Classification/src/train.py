from ast import mod
import gc
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from dataloader import LibriDataset
from model import Classifier
from utils import parse_config, preprocess_data, same_seeds, set_logger


def train(config, model, train_loader, val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = set_logger("training", config["log_path"])
    logger.info(f"Using {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    best_acc = 0.0
    for epoch in range(config["num_epoch"]):
        train_acc, train_loss = 0.0, 0.0
        val_acc, val_loss = 0.0, 0.0

        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        if len(val_set) > 0:
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)

                    loss = criterion(outputs, labels)

                    _, val_pred = torch.max(outputs, 1)
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                    val_loss += loss.item()

                logger.info(
                    "[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}".format(
                        epoch + 1,
                        config["num_epoch"],
                        train_acc / len(train_set),
                        train_loss / len(train_loader),
                        val_acc / len(val_set),
                        val_loss / len(val_loader),
                    )
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), config["model_path"])
                    logger.info(
                        "saving model with acc {:.3f}".format(best_acc / len(val_set))
                    )
        else:
            logger.info(
                "[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}".format(
                    epoch + 1,
                    config["num_epoch"],
                    train_acc / len(train_set),
                    train_loss / len(train_loader),
                )
            )


if __name__ == "__main__":
    config = parse_config()
    concat_nframes = config["concat_nframes"]
    input_dim = 39 * concat_nframes
    train_ratio = config["train_ratio"]
    train_X, train_y = preprocess_data(
        split="train",
        feat_dir="./libriphone/feat",
        phone_path="./libriphone",
        concat_num=concat_nframes,
        train_ratio=train_ratio,
    )
    val_X, val_y = preprocess_data(
        split="val",
        feat_dir="./libriphone/feat",
        phone_path="./libriphone",
        concat_num=concat_nframes,
        train_ratio=train_ratio,
    )

    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)

    del train_X, train_y, val_X, val_y
    gc.collect()

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    same_seeds(config["seed"])
    model = Classifier(
        hidden_layers=config["hidden_layers"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout_rate"],
    )
    train(config, model, train_loader, val_loader)
