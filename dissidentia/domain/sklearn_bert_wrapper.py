
"""Perform bert type model for dissident detection and wrappe it like scikit-learn"""
import os
import warnings
import numpy as np
import pandas as pd
import torch
import logging

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from dissidentia.infrastructure.dataset import get_train_test_split
from dissidentia.infrastructure.grand_debat import get_rootdir

warnings.simplefilter(action="ignore", category=FutureWarning)


class BaseBertTypeEstimator(BaseEstimator):
    """
    Base Class for Bert type Classifiers
    Parameters
    ----------
    name_model : string, default = 'camembert-base'
        path file containing huggingface pre-trained model of type bert to load
    learning_rate :float, default = 3e-5
        inital learning rate for Optimizer
    random_state : intt
        seed to initialize numpy and torch random number generators
    num_labels : int, default = 2
        number of labels for classification
    weight_decay : float, default = 0.
        regularization technique by adding a small penalty
    tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
    args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
    data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise.

    """

    MODEL_NAME = 'camembert-base'
    LOG_DIR = os.path.join(get_rootdir(), "logs")
    OUTPUT_DIR = os.path.join(get_rootdir(), "data/outputs")

    def __init__(
        self,
        name_model=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        num_labels=2,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        logging_first_step=True,
        freezing=False,
        weight_decay=0.,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_strategy="epoch",
        val_dataset = None
    ):

        self.name_model = name_model
        self.freezing = freezing
        self.num_labels = num_labels
        self.weight_decay = weight_decay
        self.save_strategy = save_strategy
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.evaluation_strategy = evaluation_strategy
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.num_train_epochs = num_train_epochs
        self.logging_first_step = logging_first_step
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.tokenizer = None
        self.model = None
        self.val_dataset = val_dataset

        self.args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=self.logging_dir,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            logging_first_step=self.logging_first_step,
            weight_decay=self.weight_decay,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model
        )

        self._validate_hyperparameters()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.name_model, num_labels=self.num_labels)
        if self.freezing:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name_model, use_fast=True)


    def _validate_hyperparameters(self):
        """
        Check hyperpameters are within allowed values.
        """

        if (not isinstance(self.num_train_epochs, int) or self.num_train_epochs < 1):
            raise ValueError(
                f"num_train_epochs must be an integer >= 1, got {self.num_train_epochs}")

        if (not isinstance(self.per_device_train_batch_size, int) or self.per_device_train_batch_size < 1):
            raise ValueError(
                f"train_batch_size must be an integer >= 1, got {self.per_device_train_batch_size}")

        if (not isinstance(self.per_device_eval_batch_size, int) or self.per_device_eval_batch_size < 1):
            raise ValueError(
                f"eval_batch_size must be an integer >= 1, got {self.per_device_eval_batch_size}")

        if self.learning_rate < 0 or self.learning_rate >= 1:
            raise ValueError(
                "learning_rate must be >= 0 and < 1, got {self.learning_rate}")

    def _compute_metrics(self, pred):
        """

        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def fit(self, x_train, y_train):
        """
        Finetune pretrained Bert model.
        Parameters
        ----------
        X : array-like of strings
            Input text
        y : 1D of strings
            Labels/targets for text data

        """
        # validate params
        self._validate_hyperparameters()


        if self.val_dataset is None:
            raise ValueError("Evaluation requires a val_dataset")
        val = pd.concat([self.val_dataset[0], self.val_dataset[1]], axis=1)
        val["labels"] = val['label'].map({True: 1, False: 0})
        df_val = Dataset.from_pandas(val, preserve_index=False)
        encoded_val = df_val.map(self._tokenize_batch, batched=True)
        encoded_val = encoded_val.remove_columns(["text", "label"])
        encoded_val.set_format("torch")
        train = pd.concat([x_train, y_train], axis=1)
        train["labels"] = train['label'].map(
            {True: 1, False: 0})
        df_train = Dataset.from_pandas(train, preserve_index=False)
        encoded_train = df_train.map(self._tokenize_batch, batched=True)
        encoded_train = encoded_train.remove_columns(["text", "label"])
        encoded_train.set_format("torch")


        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            self.model,
            self.args,
            train_dataset=encoded_train,
            eval_dataset=encoded_val,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )

        trainer.train()

        return self

    def _tokenize_batch(self, samples):
        return self.tokenizer(samples["text"], truncation=True)


class BertTypeClassifier(BaseBertTypeEstimator, ClassifierMixin):
    """
    A text classifier built on top or no of a pretrained Bert type model.
    """

    def predict_proba(self, x_test):
        """
        Make class probability predictions.
        Parameters
        ----------
        X : array-like of strings
            Input text
        Returns
        ----------
        probs: numpy 2D array of floats
            probability estimates for each class

        """
        check_is_fitted(self, ["model", "trainer"])

        df_test = Dataset.from_pandas(
            x_test.to_frame('text'), preserve_index=False)
        encoded_test = df_test.map(self._tokenize_batch, batched=True)
        encoded_test = encoded_test.remove_columns("text")
        encoded_test.set_format("torch")

        trainer = Trainer(self.model, tokenizer=self.tokenizer)

        out = trainer.predict(encoded_test)
        out_torch = torch.from_numpy(out.predictions)
        pred_prob = torch.softmax(out_torch , -1).squeeze()

        return pred_prob.numpy()

    def predict(self, x_test):
        """
        Predict most probable class.
        Parameters
        ----------
        X : array-like of strings
            Input text
        Returns
        ----------
        y_pred: numpy array
            predicted class estimates

        """
        check_is_fitted(self, ["model", "trainer"])

        y_prob = self.predict_proba(x_test)
        y_pred = np.argmax(y_prob, axis=-1)

        return y_pred.astype(bool)

    def evaluate(self):
        """ evaluate model for different metrics"""
        return Trainer(self.model, tokenizer=self.tokenizer).evaluate()

    def _tokenize_batch(self, samples):
        return self.tokenizer(samples["text"], truncation=True)
