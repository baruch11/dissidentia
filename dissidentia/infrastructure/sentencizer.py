"""Sentensizer used in the labelling and model processes """
import re
from nltk import tokenize


def sentencize(text):
    """Split text into sentences
    Parameters
    ----------
        text (string): input text

    Returns
    -------
        sentences (list of str):  input text splitted in sentences
    """
    text = re.sub(r"\.+", ".", text)  # squash many . in a unique .

    # add space after '.[A-Z]' to help nltk
    text = re.sub(r"\.([A-Z])", r". \1", text)

    return tokenize.sent_tokenize(text)

