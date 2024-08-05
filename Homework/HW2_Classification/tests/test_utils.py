from copy import deepcopy
import unittest
from torch import Tensor
import torch
from src import utils


class UtilsTest(unittest.TestCase):
    def test_load_feature(self):
        path = "./libriphone/feat/train/19-198-0008.pt"
        feature = utils.load_feature(path)
        assert isinstance(feature, Tensor)
        assert feature.shape[1] == 39

    def test_repeat(self):
        path = "./libriphone/feat/train/19-198-0008.pt"
        feature: Tensor = utils.load_feature(path)
        original_feature_num: int = feature.shape[1]
        feature = feature.repeat(1, 3)
        assert len(feature.shape) == 2
        assert feature.shape[1] == original_feature_num * 3

    def test_rotate(self):
        concat_num: int = 3
        x: torch.Tensor = torch.rand(2, 3)
        seq_len, feature_dim = x.shape[0], x.shape[1]
        x = x.repeat(1, concat_num)
        x = x.view(seq_len, concat_num, feature_dim).permute(1, 0, 2)
        first: torch.Tensor = x[0][0]
        last: torch.Tensor = x[2][-1]
        assert first.shape == (3, )
        x[0] = utils.rotate_right(x[0], 1)
        assert torch.all(torch.eq(x[0][1], first))
        x[2] = utils.rotate_left(x[2], 1)
        assert torch.all(torch.eq(last, x[2][1]))

    def test_preprocess_data(self):
        X, y = utils.preprocess_data("train", "./libriphone/feat", "./libriphone", 1)
        assert X.shape[-1] == 39
        assert len(X) == len(y)
        
        X: torch.Tensor = utils.preprocess_data("test", "./libriphone/feat", "./libriphone", 11)
        assert X.shape[-1] == 39 * 11
        


