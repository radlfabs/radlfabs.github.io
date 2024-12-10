import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, SplineTransformer
from xgboost import XGBRegressor


ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=2,
    test_size=6
    )

class QuarterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date'):
        self.date_column = date_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        dates = X[self.date_column]
        dates = pd.to_datetime(dates)

        quarter_columns = []
        quarter_conditions = [
            (dates.dt.month <= 3),
            ((dates.dt.month > 3) & (dates.dt.month <= 6)),
            ((dates.dt.month > 6) & (dates.dt.month <= 9)),
            (dates.dt.month > 9)
        ]
        quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
        
        for name, condition in zip(quarter_names, quarter_conditions):
            quarter_columns.append(condition.astype(int))

        # Combine all new features
        new_features = np.column_stack(quarter_columns)
        

        return new_features
    
    def get_feature_names_out(self, input_features=None):
        return ['Q1', 'Q2', 'Q3', 'Q4']


class MonthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='date'):
        self.date_column = date_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        dates = X[self.date_column]
        dates = pd.to_datetime(dates)

        month_column = dates.dt.month
        
        # Combine all new features
        new_features = month_column.to_numpy().reshape(-1, 1)
        
        return new_features
    
    def get_feature_names_out(self, input_features=None):
        return ['month']


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

class SinusTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.sin(X * np.pi / 12)
    
    def get_feature_names_out(self, input_features=None):
        return ['sin_month']


def make_feature_transformer(X):
    numerical_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

    spline_pipeline = Pipeline(
        steps=[
            ('month_transformer', MonthTransformer()),
            ('cyclic_hour', periodic_spline_transformer(12, n_splines=6))
        ]
    )

    sin_pipeline = Pipeline(
        steps=[
            ('month_transformer', MonthTransformer()),
            ('sine_transformer', SinusTransformer())
        ]
    )

    col_transformer = ColumnTransformer(
        transformers=[
            ('quarter_transformer', QuarterTransformer(), ['date']),
            ('month_transformer', MonthTransformer(), ['date']),
            ('sin_transformer', sin_pipeline, ['date']),
            ('spline_transformer', spline_pipeline, ['date']),
            ('num_scale', MinMaxScaler(), numerical_columns),
            ('cat_encode', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    return col_transformer


def objective(trial, X_train, y_train):
    col_transformer = make_feature_transformer(X_train)

    xgbr = XGBRegressor(
            n_estimators=500,
            learning_rate=trial.suggest_float("eta", 0.0001, 1),
            gamma=trial.suggest_int('gamma', 0, 1000),
            max_depth=trial.suggest_int("max_depth", 1, 50),
            min_child_weight=trial.suggest_int('min_child_weight', 0, 100),
            max_delta_step=trial.suggest_int('max_delta_step', 0, 100),
            subsample=trial.suggest_float('subsample', 0, 1),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0, 1),
            reg_alpha=trial.suggest_int('reg_alpha', 0, 1000),
            reg_lambda=trial.suggest_int('reg_lambda', 0, 1000),
            enable_categorical=True,
            random_state=42,
        )

    pipeline = make_pipeline(col_transformer, xgbr)

    cv_score = cross_val_score(
        pipeline,
        X=X_train,
        y=y_train,
        cv=ts_cv,
        scoring="neg_root_mean_squared_error",
    )

    return -cv_score.mean()