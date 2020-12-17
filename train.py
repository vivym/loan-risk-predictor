import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch
from torch import nn

random_seed = 0


def load_data(pca_on=False):
    dataset = np.load("./data/train.npz")
    features, targets = dataset["features"], dataset["targets"]

    if pca_on:
        pca = PCA(n_components=64, random_state=random_seed)
        pca.fit(features)
        # print("explained_variance_ratio:", pca.explained_variance_ratio_)
        features = pca.transform(features)
    print("features:", features.shape)

    train_indices = np.random.choice(targets.shape[0], math.floor(targets.shape[0] * 0.8), replace=False)
    train_mask = np.zeros((targets.shape[0],), dtype=np.bool)
    train_mask[train_indices] = 1
    train_features = features[train_mask]
    train_targets = targets[train_mask]
    test_features = features[~train_mask]
    test_targets = targets[~train_mask]

    pos_mask = train_targets == 1.
    extra_features = np.tile(train_features[pos_mask], (100, 1))
    extra_targets = np.ones((extra_features.shape[0],))

    train_features = np.concatenate((train_features, extra_features), axis=0)
    train_targets = np.concatenate((train_targets, extra_targets), axis=0)

    order = np.random.permutation(train_targets.shape[0])
    train_features, train_targets = train_features[order], train_targets[order]

    return train_features, train_targets, test_features, test_targets


class Classifier(nn.Module):
    def __init__(self, in_channels):
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)


def train(features, targets):
    print("start training.")

    clf = LogisticRegression(max_iter=10000, random_state=random_seed)
    clf.fit(features, targets)

    return clf


def main():
    np.random.seed(random_seed)

    train_features, train_targets, test_features, test_targets = load_data(pca_on=True)

    model = train(train_features, train_targets)
    pred = model.predict(test_features)
    print(pred.sum())

    correct_mask = pred == test_targets
    recall = test_targets[correct_mask].sum() / test_targets.sum()
    print("recall:", format(recall, "0.4f"))


if __name__ == "__main__":
    main()
