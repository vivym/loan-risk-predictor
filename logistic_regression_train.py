import math

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

random_seed = 0


def load_data(pca_on=False):
    dataset = np.load("./data/train.npz")
    features, targets = dataset["features"], dataset["targets"]

    if pca_on:
        pca = PCA(n_components=32, random_state=random_seed)
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
    extra_features = np.tile(train_features[pos_mask], (10, 1))
    extra_targets = np.ones((extra_features.shape[0],))

    train_features = np.concatenate((train_features, extra_features), axis=0)
    train_targets = np.concatenate((train_targets, extra_targets), axis=0)

    order = np.random.permutation(train_targets.shape[0])
    train_features, train_targets = train_features[order], train_targets[order]

    return train_features, train_targets, test_features, test_targets


def train(features, targets):
    print("start training.")
 
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, C=1., tol=1e-4, random_state=random_seed)
        # SVC(max_iter=1000, probability=True, random_state=random_seed)
    )
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
    print("recall:", format(recall * 100, "0.2f"))
    print("acc:", format(accuracy_score(pred, test_targets) * 100, "0.2f"))

    pred_prob1 = model.predict_proba(test_features)[:, 1]
    fpr, tpr, _ = roc_curve(test_targets, pred_prob1)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:", format(roc_auc * 100, "0.2f"))


if __name__ == "__main__":
    main()
