"""test Bert type model with sklearn wrapping"""

from dissidentia.infrastructure.dataset import get_train_test_split
from dissidentia.domain.sklearn_bert_wrapper import BertTypeClassifier


def camembert_sklearn_wrapper():
    """ launch with 'pytest -o python_functions="camembert_*" -s'
    """

    x_train, x_test, y_train, y_test = get_train_test_split(from_doccano=False)

    x_val = (x_test, y_test)
    model = BertTypeClassifier(val_dataset=x_val)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    print(f"test different scores: {model.evaluate()}")
