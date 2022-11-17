"""Baseline model for dissident detection"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class baselineModel(Pipeline):
    """Baseline model for dissident detection"""
    def __init__(self):
        super().__init__([
            ('tfidf', TfidfVectorizer()),
            ('lr', LogisticRegression())
        ])
