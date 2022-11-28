"""Bert model for dissident detection"""
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score)
import tensorflow as tf


class BertModelTF:
    """Bert model in tensorflow for text binary classification.
    Parameters
    ----------
        freeze_bert (bool): freeze all bert layers during fir
        epochs (int): number of epochs of training
        learning_rate (float): learnin rate of (Adam) optimizer
    """
    HUB_MODEL = "camembert-base"
    BATCH_SIZE = 8

    def __init__(self, freeze_bert=False, epochs=5, learning_rate=5e-6):
        self.tokenizer = AutoTokenizer.from_pretrained(self.HUB_MODEL)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.HUB_MODEL, num_labels=2)
        if freeze_bert:
            self.model.layers[0].trainable = False

        self.model.compile(optimizer=Adam(learning_rate),
                           metrics=['accuracy'])
        self.epochs = epochs

    def fit(self, X_train, y_train, val_data=None):
        """ fit the model
        Parameters
        ----------
            X (pd.Series(str)) : input text
            y (pd.Series(boolean)) : labels
            val_data (tuple(pd.DataFrame, pd.Series))
        """
        train_tfds = self._convert_tf_dataset(X_train, y_train)
        test_tfds = None
        if val_data:
            X_test, y_test = val_data
            test_tfds = self._convert_tf_dataset(X_test, y_test)
        self.model.fit(train_tfds, epochs=self.epochs,
                       validation_data=test_tfds)

    def predict(self, X):
        """Make predictions
        Returns
        -------
            np.array(int)
        """
        tfds = self._convert_tf_dataset(X)
        pred = np.argmax(self.model.predict(tfds).logits, axis=1)
        return pred

    def scores(self, X, y):
        metrics = [f1_score, accuracy_score, precision_score, recall_score]
        y_pred = self.predict(X)
        return {metric.__name__: metric(y, y_pred) for metric in metrics}

    def _convert_tf_dataset(self, X, y=None):
        """ convert in tf.data.Dataset
        Parameters
        ----------
            X (pd.DataFrame)
            y (pd.Series)
        Returns
        -------
            tf.data.Dataset
        """
        tok = self.tokenizer(X.to_list(), return_tensors="np",
                             truncation=True, padding=True)
        labels = np.ones(len(X))
        if y is not None:
            labels = (y*1).values
        tfds = tf.data.Dataset.from_tensor_slices(
            (dict(tok), labels))
        tfds = tfds.batch(self.BATCH_SIZE)
        return tfds


if __name__ == "__main__":  # to be deleted
    from dissidentia.infrastructure.dataset import get_train_test_split
    X_train, X_test, y_train, y_test = get_train_test_split(from_doccano=False)
    Ns = 32
    X_train = X_train.iloc[:Ns]
    y_train = y_train.iloc[:Ns]*1

    bmtf = BertModelTF(freeze_bert=False, epochs=1)
    bmtf.fit(X_train, y_train, val_data=(X_test, y_test))

    perfs = pd.DataFrame([bmtf.scores(X_test, y_test),
                          bmtf.scores(X_train, y_train)],
                         index=["test", "train"])

    with pd.option_context('display.float_format', '{:0.2f}'.format):
        print(perfs)
