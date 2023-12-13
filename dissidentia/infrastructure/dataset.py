"""Load train dataset for dissident detection """

import os
import logging
import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split

from dissidentia.infrastructure.grand_debat import get_rootdir
from dissidentia.infrastructure.doccano import DoccanoDataset


def load_dataset(from_doccano: bool = False) -> Tuple[pd.Series, pd.Series]:
    """Return tuple of sentence/label (dissident or not dissident)."""
    dataset = pd.read_csv(os.path.join(get_rootdir(), "data/labels_v4.csv"))
    if from_doccano:
        dds = DoccanoDataset()
        dataset = dds.load_data()

    dataset = dataset.loc[(dataset.label != "inclassable")]
    target = dataset.label.apply(lambda x: x == "dissident")
    sentence = dataset.text

    return sentence, target


def get_train_test_split(from_doccano=False):
    """Return dataset split used in application
    Parameters
    ----------
        from_doccano (bool): load data from doccano
    Returns:
    --------
        X_train (DataFrame)
        X_test (DataFrame),
        y_train (Series)
        y_test (Series)
    """
    X, y = load_dataset(from_doccano)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info(
        f"Split dataset\n"
        f" - train: {len(X_train)}"
        f" samples ({y_train.mean()*100:.1f}% positive)"
        "\n"
        f" - test: {len(X_test)}"
        f" examples ({y_test.mean()*100:.1f}% positive)"
    )

    return X_train, X_test, y_train, y_test
