
"""Perform bert type model for dissident detection and wrappe it like scikit-learn"""
import os
import warnings
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from dissidentia.infrastructure.grand_debat import get_rootdir

warnings.simplefilter(action="ignore", category=FutureWarning)


class BaseBertTypeEstimator(BaseEstimator):
    """
    Base Class for Bert type Classifiers
    Parameters
    ----------
    name_model : string, default = 'camembert-base'
        path file containing huggingface pre-trained model of type bert to load
    output_dir:
            output directory where the model predictions and checkpoints will be written.
    learning_rate :float, default = 3e-5
        inital learning rate for Optimizer
    per_device_train_batch_size; defaults = 8
            batch size per GPU/TPU core/CPU for training.
    per_device_eval_batch_size, defaults, = 8:
            The batch size per GPU/TPU core/CPU for evaluation.
    num_labels : int, default = 2
        number of labels for classification
    val_dataset = (x_val, y_val) : 
        dataset to evaluate the model
    weight_decay: 
        The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW
            optimizer type.
    logging_first_step, default:
            Whether to log and evaluate the first `global_step` or not.
    freezing:
            freezing or no the pre-trained model.
    num_train_epochs:
            number of epoch for the train 
            the evaluation with or without the prefix `"eval_"`.
    metric_for_best_model:
            specify the metric to use. Must be the name of a metric returned by
            the evaluation with or without the prefix `"eval_"`.
    save_strategy :
            The checkpoint save strategy to adopt during training. Possible values are:
                - "no": No save is done during training.
                - "epoch": Save is done at the end of each epoch.
                - "steps": Save is done every `save_steps.
    tokenizer:
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
    data_collator:
            The function to use to form a batch from a list of elements of `train_dataset` or `val_dataset`.

    """

    LOG_DIR = os.path.join(get_rootdir(), "logs")
    OUTPUT_DIR = os.path.join(get_rootdir(), "data/outputs")


    def __init__(
        self,
        name_model: str = 'camembert-base',
        output_dir: str = OUTPUT_DIR,
        logging_dir: str = LOG_DIR,
        num_labels: int = 2,
        evaluation_strategy: str = "epoch",
        learning_rate: float = 3e-5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        num_train_epochs: int = 10,
        logging_first_step: bool = True,
        freezing: bool = False,
        weight_decay: float = 0.,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = 'f1',
        save_strategy: str = "epoch",
        val_dataset=None
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

        encoded_val = self._torch_encode(self.val_dataset[0],
                                         self.val_dataset[1])
        encoded_train = self._torch_encode(x_train, y_train)

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

    def _torch_encode(self, X, y):
        val = pd.concat([X, y], axis=1)
        val["labels"] = val['label'].map({True: 1, False: 0})
        df_val = Dataset.from_pandas(val, preserve_index=False)
        encoded_val = df_val.map(self._tokenize_batch, batched=True)
        encoded_val = encoded_val.remove_columns(["text", "label"])
        encoded_val.set_format("torch")
        return encoded_val


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
        check_is_fitted(self, ["model"])

        df_test = Dataset.from_pandas(
            x_test.to_frame('text'), preserve_index=False)
        encoded_test = df_test.map(self._tokenize_batch, batched=True)
        encoded_test = encoded_test.remove_columns("text")
        encoded_test.set_format("torch")

        trainer = Trainer(self.model, tokenizer=self.tokenizer)

        out = trainer.predict(encoded_test)
        out_torch = torch.from_numpy(out.predictions)
        pred_prob = torch.softmax(out_torch, -1).squeeze()

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
        check_is_fitted(self, ["model"])

        y_prob = self.predict_proba(x_test)
        y_pred = np.argmax(y_prob, axis=-1)

        return y_pred.astype(bool)

    def evaluate(self):
        """ evaluate model for different metrics"""
        X, y = self.val_dataset
        return Trainer(self.model,
                       tokenizer=self.tokenizer,
                       eval_dataset=self._torch_encode(X, y),
                       ).evaluate()
