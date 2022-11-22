"""Opitmization of baseline model """
import pandas as pd

from dissidentia.infrastructure.dataset import get_train_test_split
from dissidentia.domain.baseline_model import baselineModel


def optimize_baseline():
    """ Full display for baseline mode optimization debug
    launch with 'pytest -o python_functions="optimize_*" -s'
    """
    X_train, X_test, y_train, y_test = get_train_test_split(from_doccano=True)
    blm = baselineModel()

    blm.fit(X_train, y_train)

    cv_res = pd.DataFrame(blm.cv_results_)
    for rm_suff in ["param_lr__", "param_tfidf__"]:
        cv_res = cv_res.rename(columns={col: col[len(rm_suff):]
                                        for col in cv_res.columns
                                        if rm_suff in col})
    cv_res = cv_res.sort_values(by="mean_test_score", ascending=False)
    with pd.option_context('display.float_format', '{:0.2f}'.format):
        print(cv_res[['min_df', 'max_df', 'C', 'penalty', 'mean_test_score',
                      'mean_train_score']])

    print(f"test f1 score: {blm.score(X_test, y_test):.2f}")
