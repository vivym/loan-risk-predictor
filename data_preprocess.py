import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno


def preprocess_number(col):
    null_mask = col.isnull()
    num_null = null_mask.sum()
    if num_null / col.shape[0] > 0.6:
        return None
    
    # TODO: find a better way.
    col[null_mask] = col[~null_mask].mean()

    feature = np.asarray(col, dtype=np.float64)
    mx, mn = feature.max(), feature.min()
    feature = feature * 2. / (mx - mn) - 1.
    return feature


def preprocess_str(col):
    null_mask = col.isnull()
    num_null = null_mask.sum()
    if num_null / col.shape[0] > 0.6:
        return None

    col[null_mask] = col[~null_mask].mode()[0]

    items = pd.unique(col)
    for idx, s in enumerate(items):
        col[col == s] = str(idx)
    col = pd.to_numeric(col)
    feature = np.asarray(col, dtype=np.float64)
    feature = feature * 2. / len(items) - 1.
    return feature


def main():
    train_data = pd.read_csv("./data/application_train.csv")
    num_train_data = train_data.shape[0]
    test_data = pd.read_csv("./data/application_test.csv")
    num_test_data = test_data.shape[0]
    print(test_data.shape)
    data = train_data.append(test_data)
    print(num_train_data, num_test_data)

    features, targets = [], None
    for col in data.columns:
        if col in ["SK_ID_CURR"]:
            continue
        if col == "TARGET":
            targets = np.asarray(data[col][:num_train_data])
        else:
            num_null = data[col].isnull().sum()
            if num_null > 0:
                print(col, num_null, format((100 * num_null / data[col].shape[0]), "0.2f"))
            if data[col].dtype == object:  # str
                feature = preprocess_str(data[col].copy())
                
            else:  # int64 or float64
                feature = preprocess_number(data[col].copy())

            if feature is not None:
                features.append(feature)

    features = np.stack(features, axis=0).T
    
    train_features = features[:num_train_data]
    test_features = features[num_train_data:]

    np.savez_compressed("./data/train.npz", features=train_features, targets=targets)
    np.savez_compressed("./data/test.npz", features=test_features)


if __name__ == "__main__":
    main()
