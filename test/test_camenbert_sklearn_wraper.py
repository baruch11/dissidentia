"""test Bert type model with sklearn wrapping"""

import numpy as np
from dissidentia.infrastructure.dataset import get_train_test_split
from dissidentia.domain.sklearn_bert_wrapper import BertTypeClassifier
from dissidentia.domain.model_wrapper import DissidentModelWrapper
import tempfile

def camembert_sklearn_wrapper():
    """ launch with 'pytest -o python_functions="camembert_*" -s'
    """

    x_train, x_test, y_train, y_test = get_train_test_split(from_doccano=False)
    x_train = x_train.iloc[:32]
    y_train = y_train.iloc[:32]
    x_test = x_test.iloc[:32]
    y_test = y_test.iloc[:32]

    x_val = (x_test, y_test)
    model = BertTypeClassifier(val_dataset=x_val,
                               num_train_epochs=1, freezing=True)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    print(f"test different scores: {model.evaluate()}")

    # test save / load
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)

        model2 = BertTypeClassifier.load(tmpdir)
        y_proba2 = model2.predict_proba(x_test)

        assert np.max(np.abs(y_prob-y_proba2)) < 1e-6


def test_camembert_dissident_wrapper():
    x_train, x_test, y_train, y_test = get_train_test_split(from_doccano=False)
    x_train = x_train.iloc[:32]
    y_train = y_train.iloc[:32]
    x_test = x_test.iloc[:32]
    y_test = y_test.iloc[:32]

    x_val = (x_test, y_test)

    model = BertTypeClassifier(val_dataset=x_val,
                               num_train_epochs=1, freezing=True)

    DissidentModelWrapper(model).sentensize_and_predict("Hello.")

    # model_load = DissidentModelWrapper.load("BertTypeClassifier")

    # proba1 = model.predict_proba(x_test.values)
    # proba2 = model_load.sentensize_and_predict("Hello.")

    # print(proba2)
    #assert np.max(np.abs(proba1-proba2)) < 1e-6
