from mylib.train import cv_parameters, Trainer, SyntheticBernuliDataset
from sklearn.linear_model import LogisticRegression

def test_sample():
    assert 0 == 0


def test_dataset():
    dataset = SyntheticBernuliDataset(n=10, m=100, seed=42)

    assert len(dataset.X) == len(dataset.y)

def test_trainer():
    trainer = Trainer(
        LogisticRegression(penalty='l1', solver='saga', C=1.0),
        dataset.X, dataset.y,
    )

    trainer.train()
    assert trainer.eval(output_dict=True)['accuracy'] == 0.96

def test_cv():
    Cs, accuracy, parameters = cv_parameters(dataset.X, dataset.y)

    assert len(Cs) == len(accuracy) == len(parameters)