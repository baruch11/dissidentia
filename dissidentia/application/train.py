"""Compute training from a csv file"""

import logging
import argparse
import pandas as pd


from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from dissidentia.domain.baseline_model import baselineModel
from dissidentia.domain.sklearn_bert_wrapper import BertTypeClassifier
from dissidentia.infrastructure.dataset import get_train_test_split
from dissidentia.domain.model_wrapper import DissidentModelWrapper

logging.getLogger().setLevel(logging.INFO)

MODELS = [BertTypeClassifier, baselineModel]
MODEL_NAMES = ", ".join([model.__name__ for model in MODELS])

PARSER = argparse.ArgumentParser(
    description='Compute training from a csv file')
PARSER.add_argument('--debug', '-d', action='store_true',
                    help='activate debug logs')
PARSER.add_argument('--doccano', '-dc', action='store_true',
                    help='activate debug logs')
PARSER.add_argument('--save_model', '-s', action='store_true',
                    help='save model in a pickle')
PARSER.add_argument('--model', '-m', default=MODELS[0].__name__,
                    help=f'model_name in ({MODEL_NAMES})')
PARSER.add_argument('--no_fit', '-n', action='store_true',
                    help='just load and evaluate the model')

args = PARSER.parse_args()

if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)

X_train, X_test, y_train, y_test = get_train_test_split(
    from_doccano=args.doccano)

for constructor in MODELS:
    if args.model == constructor.__name__:
        break

if args.no_fit:
    model = DissidentModelWrapper.load(args.model).model
else:
    model = constructor()
    model.fit(X_train, y_train)


y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

metrics = [accuracy_score, f1_score, precision_score, recall_score]

perfs = pd.DataFrame(
    {metric.__name__: {"train": metric(y_train, y_pred_train),
                       "test": metric(y_test, y_pred_test)}
     for metric in metrics})

with pd.option_context('display.float_format', '{:0.2f}'.format):
    print(f"Performances:\n{perfs}")

# save model
if args.save_model:
    DissidentModelWrapper(model).save()
