"""Baseline model for dissident detection"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

class baselineModel(RandomizedSearchCV):
    """Baseline model for dissident detection"""
    PARAM_DIST = {"lr__C": uniform(loc=1, scale=20),
                  "lr__penalty": ['l2', 'l1'],
                  'tfidf__max_df': uniform(loc=.8, scale=.2),
                  'tfidf__min_df': list(range(10)),
                  }


    def __init__(self):
        self.pipe = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('lr', LogisticRegression(solver='liblinear', class_weight='balanced',
                                      tol=1e-4, max_iter=100))
        ])
        super().__init__(self.pipe, self.PARAM_DIST, random_state=42,
                         verbose=True, n_jobs=-1, cv=3, scoring='f1',
                         n_iter=50, return_train_score=True
                         )
