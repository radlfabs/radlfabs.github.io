import json
import logging
import os

import joblib
import numpy as np
import optuna
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import plotly.express as px
from mapie.regression import MapieTimeSeriesRegressor
from mapie.metrics import (regression_coverage_score,
                           regression_mean_width_score)
from mapie.subsample import BlockBootstrap


from pipeline import make_feature_transformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SimpleModel():
    def __init__(self, fixed_params=None, cache_dir='optimization_cache'):
        if fixed_params is None:
            fixed_params = {
                "n_estimators": 500,
                "enable_categorical": True,
                "random_state": 42
            }
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.fixed_params = fixed_params
        self.is_fitted = False

    def set_objective(self, objective):
        self.objective = objective

    def set_fixed_params(self, fixed_params):
        self.fixed_params = fixed_params

    def set_studies(self, studies):
        self.studies = studies

    def _make_writable_string(self, location):
        return (
            location
            .lower()
            .replace(".", "")
            .replace(" ", "_")
            .replace("ü", "ue")
            .replace("ö", "oe")
            .replace("ä", "ae")
            .replace("ß", "ss")
        )

    def _get_cache_path(self, location):
        """Generate a cache file path for a specific location."""
        return os.path.join(self.cache_dir, f"{location}_study_cache.json")
    
    def _save_study_cache(self, location, study_data):
        """Save study results to a JSON cache file."""
        cache_path = self._get_cache_path(location)
        with open(cache_path, 'w') as f:
            json.dump(study_data, f)
    
    def _load_study_cache(self, location):
        """Load cached study results for a location."""
        cache_path = self._get_cache_path(location)
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def _save_failed_cache(self, failed_studies):
        """Save failed studies to a JSON cache file."""
        cache_path = os.path.join(self.cache_dir, "failed_studies.json")
        with open(cache_path, 'w') as f:
            json.dump(failed_studies, f)
    
    def _load_failed_cache(self):
        """Load failed studies from a JSON cache file."""
        cache_path = os.path.join(self.cache_dir, "failed_studies.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return {}

    def tune(self, train_data, n_trials=200, use_stored=False):
        metrics = {}
        succesful_studies = {}
        failed_studies = {} if not use_stored else self._load_failed_cache()

        for location, group in train_data.group_by('location'):
            location = location[0]

            logger.info(f"Tuning {location}.")

            if location in failed_studies.keys():
                continue
            
            loc_str = self._make_writable_string(location)
            # Check if we can use stored results
            if use_stored:
                cached_study = self._load_study_cache(loc_str)
                if cached_study:
                    logger.info(f"Using stored hyperparameters for {location}.")
                    inner_dict = cached_study[location]
                    succesful_studies[location] = inner_dict['best_params']
                    metrics[location] = inner_dict['best_value']
                    continue
            
            X_train = group.drop("count").to_pandas()
            y_train = group["count"]
            study = optuna.create_study(direction="minimize")
            
            try:
                logger.info(f"Optimizing hyperparameters for {location}.")
                study.optimize(
                    lambda trial: self.objective(trial, X_train, y_train), 
                    n_trials=n_trials
                    )
                succesful_studies[location] = study.best_trial.params
                metrics[location] = study.best_trial.value

                self._save_study_cache(
                    loc_str, 
                    {
                        location:
                        {
                        'best_params': study.best_trial.params,
                        'best_value': study.best_trial.value
                        }
                    }
                    )
            except ValueError as e:
                failed_studies[location] = str(e)

        self.succesful_studies = succesful_studies
        self.metrics = metrics

        # find location with the min mape in the dict
        self.best_location = pd.Series(metrics).idxmin()
        best_params = succesful_studies[self.best_location]

        for location in failed_studies.keys():
            self.succesful_studies[location] = best_params
            self._save_study_cache(
                self._make_writable_string(location),
                {
                    location: {
                        'best_params': best_params,
                        'best_value': metrics[self.best_location]
                    }
                }
            )
        self.failed_studies = failed_studies
        self._save_failed_cache(failed_studies)


    def _refit_inner(self, train_data, location):
        X_train = train_data.drop("count").to_pandas()
        y_train = train_data["count"]

        pipe = make_pipeline(
            make_feature_transformer(X_train),
            XGBRegressor(
                **self.succesful_studies[location],
                **self.fixed_params
            )
        )
        return pipe.fit(X_train, y_train)

    def refit(self, train_data):
        # first refit the baseline model
        self._baseline_pipe = self._refit_inner(train_data, self.best_location)
        self._pipeline_dict = {}
        # fit location specific models
        for location, data in self.succesful_studies.items():
            logger.info(f"Refitting model for {location}.")
            self._pipeline_dict[location] = self._refit_inner(
                train_data=train_data.filter(pl.col("location") == location),
                location=location
            )
        self.is_fitted = True

    def check_fitted(self):
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return True

    def predict(self, X_test, loc_str=""):
        self.check_fitted()
        # return the specific model when available, otherwise the baseline model
        # and make predictions
        return self._pipeline_dict.get(loc_str, self._baseline_pipe).predict(X_test)

    def eval(self, test_data):
        self.check_fitted()
        metrics_by_location = {}
        for location, group in test_data.group_by('location'):
            location = location[0]
            X_test = group.drop("count").to_pandas()
            y_test = group["count"]
            y_pred = self.predict(X_test, location)

            metrics_by_location[location] = {
                "rmse": np.round(root_mean_squared_error(y_test, y_pred)),
                "mape": np.round(mean_absolute_percentage_error(y_test, y_pred), 2)
            }

        return (
            pl.DataFrame(
                pd.DataFrame(metrics_by_location).T
                .reset_index()
                .rename(columns={"level_0": "location", "index": "location"})
            )
            .with_columns(
                pl.col("location").is_in(self.succesful_studies.keys()).not_().alias("baseline"),
                pl.col("location").cast(pl.Categorical).alias("location"),
                pl.col("rmse").cast(pl.Int64).alias("rmse"),
            )
            .join(
                test_data
                .group_by("location")
                .agg(pl.col("count").mean().cast(pl.Int64).alias("count"))
                .filter(pl.col("location").is_in(metrics_by_location.keys()))
                , on="location"
                )
            .sort("mape")
        )

    def get_pipeline(self, loc_str):
        return self._pipeline_dict.get(loc_str, self._baseline_pipe)
    
    def save(self, path="models/trained_ml_pipeline.pkl"):
        joblib.dump(self, path)

    def plot_predictions(self, data, loc_str):
        if not os.path.exists("images"):
            os.mkdir("images")

        self.check_fitted()
        X_test = data.filter(data["location"] == loc_str).drop("count").to_pandas()
        y_test = data.filter(data["location"] == loc_str)["count"].to_pandas()
        y_pred = self.predict(X_test, loc_str)

        combined = pd.DataFrame({"date": X_test["date"], "Actual": y_test, "Predicted": y_pred})

        fig = px.line(
            combined, x="date", y=["Actual", "Predicted"], 
            title=f"Counter {loc_str}", 
            template="simple_white")
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Count')
        # fig.write_image(f"images/{loc_str}.png")
        fig.show()


class MAPIEModel(SimpleModel):
    def set_succesful_studies(self, succesful_studies):
        self.succesful_studies = succesful_studies
    
    def set_best_location(self, best_location):
        self.best_location = best_location

    def _fit_mapie(self, train_data, location, n_resamplings=100, n_blocks=3):
        X_train = train_data.drop("count").filter(train_data["location"] == location).to_pandas()
        y_train = train_data.filter(train_data["location"] == location)["count"]

        pipe = make_pipeline(
                make_feature_transformer(X_train),
                XGBRegressor(
                    **self.succesful_studies[location],
                    **self.fixed_params
                )
            )

        cv_mapietimeseries = BlockBootstrap(
            n_resamplings=n_resamplings, n_blocks=n_blocks, overlapping=False, random_state=42
        )
        mapie = MapieTimeSeriesRegressor(
            pipe,
            method="enbpi",
            cv=cv_mapietimeseries,
            agg_function="mean",
            n_jobs=-1,
        )
        return mapie.fit(X_train, y_train)

    def fit_mapie(self, train_data, n_resamplings=100, n_blocks=3):
        self.refit(train_data)
        self._baseline_mapie = self._fit_mapie(train_data, location=self.best_location)

        self.mapie_models = {}
        for location in self.succesful_studies.keys():
            logger.info(f"Fitting MAPIE model for {location}.")
            self.mapie_models[location] = self._fit_mapie(train_data, location=location, n_resamplings=n_resamplings, n_blocks=n_blocks)


        self.is_mapie_fitted = True

    def check_mapie_fitted(self):
        if not hasattr(self, "is_mapie_fitted") or not self.is_mapie_fitted:
            raise ValueError("Model has not been fitted yet.")
        return True

    def predict_mapie(self, X_test, loc_str="", alpha=0.05):
        self.check_mapie_fitted()

        mapie = self.mapie_models.get(loc_str, self._baseline_mapie)
        y_pred, y_pis = mapie.predict(X_test, alpha=alpha)

        return y_pred, y_pis

    def print_mapie_metrics(self, y_test, y_pis):
        coverage = regression_coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0])
        width = regression_mean_width_score(y_pis[:, 0, 0], y_pis[:, 1, 0])
        print(
            "Coverage and prediction interval width mean for CV+: "
            f"{coverage:.3f}, {width:.3f}"
        )

    def save(self, path="models/trained_quantile_pipeline.pkl"):
        joblib.dump(self, path)

    def plot_mapie(self, data, loc_str, alpha=0.05):
        import plotly.graph_objects as go
        self.check_mapie_fitted()

        X_test = data.filter(data["location"] == loc_str).drop("count").to_pandas()
        y_test = data.filter(data["location"] == loc_str)["count"].to_pandas()
        y_pred, y_pis = self.predict_mapie(X_test, loc_str=loc_str, alpha=alpha)

        fig = go.Figure([
            go.Scatter(
                name=f'{(1 - alpha)*100}% PI',
                x=X_test['date'],
                y=y_pis[:, 1, 0],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name=f'{(1 - alpha)*100}% PI',
                x=X_test['date'],
                y=y_pis[:, 0, 0],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=True
            ),
            go.Scatter(
                x=X_test["date"],
                y=y_test,
                name="Actual",
                line=dict(color='rgb(31, 119, 180)'),
                mode='lines'
            ),
            go.Scatter(
                x=X_test["date"],
                y=y_pred,
                name="Predicted",
                line=dict(color='rgb(255, 127, 14)'),
                mode='lines'
            )
        ])
        fig.update_layout(
            title=dict(text=f"Prediction Intervals for {loc_str}"),
            hovermode="x",
            template="simple_white"
            )
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Count')
        fig.show()