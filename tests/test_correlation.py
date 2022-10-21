import torch
from semissl.losses.correlation import feature_correlation_loss


def test_feature_correlation_loss():
    X = torch.randn(125, 15)
    label = torch.randint(0, 7, (125, ))
    corr = feature_correlation_loss(X, X, label)
    assert corr > 0

def is_sorted(x):
    return all(v1 <= v2 for v1, v2 in zip(x, x[1:])) 

def test_labels_construction():
    labels = torch.randint(0, 7, (5, ))
    labels = torch.cat([labels, labels])
    labels_index = torch.argsort(labels)
    labels_unique, _ = torch.unique(labels, return_counts=True)
    index = [
        (labels[labels_index] == c).nonzero(as_tuple=True)[0][0] for c in labels_unique
    ]
    assert is_sorted(index)