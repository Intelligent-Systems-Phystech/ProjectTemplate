import mylib
from mylib.train import cv_parameters, Trainer, SyntheticBernuliDataset

def test_sample():
    assert 0 == 0


def test_dataset():
    dataset = SyntheticBernuliDataset(n=10, m=100, seed=42)

    assert len(dataset.X) == len(dataset.y)