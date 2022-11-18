"""Load train dataset for dissident detection """

import os
import logging
import pandas as pd

from sklearn.model_selection import train_test_split

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
    dataset = pd.concat([
        pd.read_csv(os.path.join(get_rootdir(), "data/labels_v1.csv")),
        pd.read_csv(os.path.join(get_rootdir(), "data/labels_v2.csv"))
    ])
    dataset = dataset.loc[(dataset.moindze != "inclassable") &
                          (dataset.amir != "inclassable") &
                          (dataset.charles != "inclassable")]
    y = dataset.final.apply(lambda x: x == "dissident")
    X = dataset.text
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    logging.info(
        f"Split dataset\n"
        f" - train: {len(X_train)} samples ({y_train.mean()*100:.1f}% pos)"
        "\n"
        f" - test: {len(X_test)} examples ({y_test.mean()*100:.1f}% pos)"
    )

    return X_train, X_test, y_train, y_test
