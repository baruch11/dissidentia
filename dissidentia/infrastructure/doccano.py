"""Python interface to Quantmetry doccano server"""

import os
import tempfile
import zipfile
import logging
import time

import pandas as pd
from tqdm import tqdm
try:
    from doccano_client import DoccanoClient
except ModuleNotFoundError:
    pass  # doccano_client not working on colab but not needed


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
            path_to_zip_file = self.client.download(
                self.project_id, format="CSV", only_approved=only_approved,
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

    def safe_upload_data(self, data, proceed=False, label=None,
                         approve=True):
        """Upload data to doccano database avoiding dupplicates
        Parameters
        ----------
            data (pd.DataFrame): dataframe with a least the columns
                text, question_id, sent_id (index of the sentences inside a
                single answer)
            proceed (boolean): proceed upload if True (else dry-run to let
                the user inspect the result)
            label (str): if not None add a 'label' column, label must be
                'dissident' or 'non dissident'
            approve (bool): approve new examples if True
        Returns
        -------
             df_upload (pd.DataFrame): the dataframe to be upload (if proceed)
        """
        print("load doccano ds")
        doccano_ds = self.load_data(only_approved=False)
        ids_orig = doccano_ds.id.values
        print("done")
        doccano_set = set(zip(doccano_ds.question_id, doccano_ds.sent_id))

        upload_set = set(zip(data.question_id, data.sent_id))
        drop_elt = upload_set.intersection(doccano_set)

        # drop potential dupplicates
        df_upload = data.loc[data.apply(
            lambda s: (s.question_id, s.sent_id) not in drop_elt,
            axis=1)]

        if label:
            assert label in ["dissident", "non dissident"]
            df_upload["label"] = label

        print(f"dropping {len(drop_elt)}"
              f", keep {len(df_upload)} examples")

        if not proceed:
            print("dry run (proceed=False), not uploading anything")
            return df_upload

        print("upload examples...")
        self.upload_data(df_upload)
        print("done")

        print("sleep 2")
        time.sleep(2)

        if label is not None and approve:
            print("Approve new examples")
            ex_iter = self.client.list_examples(self.project_id,
                                                is_confirmed=False)
            for example in tqdm(ex_iter):
                if example.id not in ids_orig:
                    self.client.update_example_state(
                        self.project_id, example.id)
            print("Approve done")

        return df_upload
