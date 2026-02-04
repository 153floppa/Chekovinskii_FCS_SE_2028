import pandas as pd
from utils import get_columns_grouped_by_dtypes, x_y_split
from config import RANDOM_SEED

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from typing import Tuple


LINEAR_MODEL = None


def train_model(df: pd.DataFrame, target_column: str, val_size=0.000001, test_size=0.2):
    global LINEAR_MODEL

    assert val_size + test_size <= 0.7, "Amount of data for train is really low"

    tmp_df, test_df = train_test_split(df, test_size=test_size, random_state=RANDOM_SEED)
    train_df, val_df = train_test_split(tmp_df, test_size=val_size * (1 - test_size), random_state=RANDOM_SEED)

    grouped_cols = get_columns_grouped_by_dtypes(df)
    train_cols = list(grouped_cols['int64']) + list(grouped_cols['float64'])

    X_train, y_train = x_y_split(train_df, target_column, use_cols=train_cols)
    X_val, y_val = x_y_split(val_df, target_column, use_cols=train_cols)
    X_test, y_test = x_y_split(test_df, target_column, use_cols=train_cols)

    LINEAR_MODEL = Ridge(random_state=RANDOM_SEED)

    LINEAR_MODEL.fit(
        X_train,
        y_train,
    )

    predicted = LINEAR_MODEL.predict(X_test)

    predicted_score = r2_score(y_test, predicted)
    print(predicted_score)
