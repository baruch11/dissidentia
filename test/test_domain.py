"""unit tests for domain code"""
import os
import numpy as np

from dissidentia.infrastructure.grand_debat import get_rootdir
from dissidentia.domain.model_wrapper import DissidentModelWrapper

class _dummy_model:
    def predict(self, corpus):
        return np.random.choice([True, False], len(corpus))


def test_model_wrapper():
    """test DissidentModelWrapper """
    print("test_model_wrapper")
    test_text = "Hello. Bien ou quoi ?"
    dummy = _dummy_model()

    # check sentensize_and_predict
    mwr = DissidentModelWrapper(dummy)
    output = mwr.sentensize_and_predict(test_text)
    print(output)
    assert len(output) == 2

    # check creation of _dummy_model.pkl
    filepath = os.path.join(get_rootdir(), "data/models/",
                            dummy.__class__.__name__+'.pkl')
    if os.path.exists(filepath):
        os.remove(filepath)
    mwr.save()
    assert os.path.exists(filepath)

    # check load
    loadmodel = DissidentModelWrapper.load("_dummy_model")
    output = loadmodel.sentensize_and_predict(test_text)
    print(f"load model predict :{output}")
    assert len(output) == 2

