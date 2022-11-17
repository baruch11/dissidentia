"""Load train dataset for dissident detection """

import os
import pandas as pd
from dissidentia.infrastructure.grand_debat import get_rootdir


def get_train_test_split():
    """Return dataset split used in application
    Returns:
    --------
        X_train (DataFrame)
        X_test (DataFrame),
        y_train (Series)
        y_test (Series)
    """
    dataset_path = os.path.join(get_rootdir(), "data/labels_v1.csv")
    labels_v1 = pd.read_csv(dataset_path)
    labels = labels_v1.final
    y_train = labels.loc[labels != "inclassable"] == "dissident"
    X_train = labels_v1.text.loc[labels != "inclassable"]

    return X_train, X_train, y_train, y_train
