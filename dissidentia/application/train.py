"""Compute training from a csv file"""

import sys
import logging
import argparse
import pandas as pd


from sklearn.metrics import accuracy_score, f1_score
from dissidentia.domain.baseline_model import baselineModel
from dissidentia.infrastructure.dataset import get_train_test_split
from dissidentia.domain.model_wrapper import DissidentModelWrapper

logging.getLogger().setLevel(logging.INFO)
PARSER = argparse.ArgumentParser(
    description='Compute training from a csv file')
PARSER.add_argument('--debug', '-d', action='store_true',
                    help='activate debug logs')

args = PARSER.parse_args()

if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)

X_train, X_test, y_train, y_test = get_train_test_split()
model = baselineModel()
model.fit(X_train, y_train)


y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

metrics = [accuracy_score, f1_score]

perfs = pd.DataFrame(
    {metric.__name__: {"train": metric(y_train, y_pred_train),
                       "test": metric(y_test, y_pred_test)}
     for metric in metrics})

with pd.option_context('display.float_format', '{:0.2f}'.format):
    print(f"Performances:\n{perfs}")

# save model
DissidentModelWrapper(model).save()
