import pandas as pd
from sklearn.model_selection import KFold


def cross_validation(
    df_train: pd.DataFrame,
    n_folds: int,
    random_state: int,
):
    kfold = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )
    df_train["fold"] = -1
    for fold, (_, valid_idx) in enumerate(kfold.split(df_train)):
        df_train.loc[valid_idx, "fold"] = fold
    return df_train
