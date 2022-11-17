"""Wrapper to provide common methods to detection models"""
import os
import pickle
import logging

from dissidentia.infrastructure.sentencizer import sentencize
from dissidentia.infrastructure.grand_debat import get_rootdir


class DissidentModelWrapper:
    """This class is a wrapper to provide common methods to detection models"""
    PICKLE_ROOT = os.path.join(get_rootdir(), "data/models")

    def __init__(self, model):
        self.model = model

    def sentensize_and_predict(self, text):
        """Split a text int sentences and predict 'dissident' label
        Parameters
        ----------
            text (str): input text

        Returns
        -------
            list of tuples (str, boolean): sentences and 'dissident' label
        """
        sentences = sentencize(text)
        predictions = self.model.predict(sentences)
        return list(zip(sentences, predictions))

    @classmethod
    def load(cls, model_name):
        """Load model from its name."""
        src_dir = os.path.join(cls.PICKLE_ROOT, model_name+".pkl")
        with open(src_dir, 'rb') as inp:
            return pickle.load(inp)

    def save(self):
        """Save the model (pkl format)."""
        pickle_path = self._get_pickle_path()
        logging.info("Saving model in %s", pickle_path)
        with open(pickle_path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        return True

    def _get_pickle_path(self):
        return os.path.join(
            self.PICKLE_ROOT, self.model.__class__.__name__)+".pkl"
