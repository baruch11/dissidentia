"""This module loads answers from the grand debat"""

import os
import logging
from dataclasses import dataclass
import hashlib
import wget
import pandas as pd


def get_rootdir():
    """Return rootdir absolute path"""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../"))


def md5(fname):
    "Return md5sum of a file."
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@dataclass
class GDAnswers:
    """Represents french citizen answers."""
    ANSWERS_URL = "https://www.data.gouv.fr/fr/datasets/r/18ce2874-16f4-4b1a-a48e-5438b263b9d5"
    MD5 = "d8ff6bddf206326ca8881a85c3fb12db"
    QUESTIONS = [
        "QUXVlc3Rpb246MTY5 - Que pensez-vous de l'organisation de l'Etat et des administrations en France ? De quelle manière cette organisation devrait-elle évoluer ?"
    ]

    csv_file: str = os.path.join(get_rootdir(), "data/answers.csv")

    def import_data(self):
        """Import answers.csv file from data gouv to 'data' directory """

        if not self._check_md5():

            def bar_custom(current, total, width=80):
                print(f"\rDownloading: {current / total * 100:.1f}%"
                      f" [{current} / {total}] bytes", end='')

            logging.info("Download %s", self.ANSWERS_URL)
            wget.download(self.ANSWERS_URL, bar=bar_custom,
                          out=self.csv_file)

        logging.info("%s imported on the disk",
                     os.path.basename(self.csv_file))

    def _check_md5(self):
        """check md5sum of local answers.csv if the file exists"""
        try:
            if md5(self.csv_file) != self.MD5:
                logging.info("md5 check failed for %s", self.csv_file)

        except FileNotFoundError:
            logging.info("%s not found!!", self.csv_file)
            return False
        return True

    def load_data(self, nrows=None):
        """Return a list of answers of the question.
        Returns
        -------
           ans (list of string): each element of the list is a document
           representing one citizen answer
        """
        self.import_data()
        ans = pd.read_csv(self.csv_file, nrows=nrows, usecols=self.QUESTIONS)\
                .dropna().iloc[:, 0].values

        return ans
