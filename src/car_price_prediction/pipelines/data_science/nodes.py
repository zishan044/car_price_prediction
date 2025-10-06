import logging
import typing as t
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

def split_data(df: pd.DataFrame, params: t.Dict) -> t.Tuple:
    X = df[params['features']]
    y = df['num__price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=params['random_state'])
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    regressor = GradientBoostingRegressor()
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_model(regressor: GradientBoostingRegressor, X_test=pd.DataFrame, y_test=pd.Series):
    y_pred = regressor.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Absolute Error: {mae}\nMean Squared Error: {mse}\nRoot Mean Squared Error: {rmse}\nR2 Score: {r2}")
    