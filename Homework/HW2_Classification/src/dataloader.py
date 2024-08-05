import torch
from torch.utils.data import Dataset


class LibriDataset(Dataset):
    data: torch.Tensor
    label: torch.Tensor | None

    def __init__(self, X: torch.Tensor, y: torch.Tensor = None) -> None:
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
