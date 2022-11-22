"""Python interface to Quantmetry doccano server"""

import os
import tempfile
import zipfile
import logging

import pandas as pd
from doccano_client import DoccanoClient



class DoccanoDataset:
    """This class represents the dissidentia doccano dataset"""
    DOCCANO_URL = 'https://qmdoccano.azurewebsites.net'

    def __init__(self):
        self.client = DoccanoClient(self.DOCCANO_URL)

        username = os.getenv('DOCCANO_LOGIN')
        password = os.getenv('DOCCANO_PASSWORD')
        if (username is None) or (password is None):
            raise RuntimeError("set your DOCCANO_LOGIN DOCCANO_PASSWORD"
                               " environment variables")

        self.client.login(username=username, password=password)

        self.project_id = None
        for project in self.client.list_projects():
            if project.name == "dissidentia":
                self.project_id = project.id
        assert self.project_id is not None

    def load_data(self, only_approved=True):
        """Load all the approved examples of dissidentia doccano dataset
        Parameters
        ----------
            only_approved (bool): load only approved sentences
        Returns
        -------
            dataset (pd.DataFrame): doccano dataset, with a least the columns
            text, label, question_id, sent_id (index of the sentences inside a
            single answer)
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            path_to_zip_file = self.client.download(self.project_id,
                                                    format="CSV",
                                                    only_approved=only_approved,
                                                    dir_name=tmpdirname)

            with zipfile.ZipFile(str(path_to_zip_file), 'r') as zip_ref:
                namelist = zip_ref.namelist()
                assert len(namelist) == 1
                fextract = zip_ref.extract(namelist[0])
                return pd.read_csv(fextract)

    def upload_data(self, data):
        """Upload data to doccano dissidentia database
        Parameters
        ----------
            data (pd.DataFrame): dataframe with a least the columns
            text, question_id, sent_id (index of the sentences inside a
            single answer)
        """
        required_cols = ["text", "question_id", "sent_id"]
        assert all(col in data.columns for col in required_cols)
        with tempfile.TemporaryDirectory() as tmpdirname:
            uploadpath = os.path.join(tmpdirname, "upload.csv")
            data.to_csv(uploadpath)
            self.client.upload(self.project_id,
                               file_paths=[uploadpath],
                               task='DocumentClassification',
                               format='CSV',
                               column_data='text',
                               column_label='label')
            logging.info("Uploaded %d datas", len(data))
