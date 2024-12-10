
import os
import dotenv
import requests
import polars as pl
from pathlib import Path
from datetime import datetime

dotenv.load_dotenv(Path(".env"))
tomorrow_api_key = os.getenv('TOMORROW_API_KEY')

location = '50.938361,6.959974'
forecast_api = "https://api.tomorrow.io/v4/weather/forecast"

params = {
    "location": location,
    "apikey": tomorrow_api_key
}

headers = {"accept": "application/json"}

r = requests.get(forecast_api, params=params, headers=headers)
r_data = r.json()

train_data = pl.read_csv("data/train_data.csv")
inference_data = (
    pl.DataFrame(r_data["timelines"]["daily"][0]["values"])
    .select(
        "temperatureMax",
        "temperatureMin",
        "temperatureAvg",
        "rainAccumulationMax",
        "windSpeedMax",
        "windSpeedAvg",
        "cloudCoverAvg"
    )
    .with_columns(
        pl.lit("Venloer Straße").alias("location"),
        pl.col("temperatureMax").alias("air_temp_daymax_month_max"),
        pl.col("temperatureMax").alias("air_temp_daymax_month_mean"),
        pl.col("temperatureMin").alias("air_temp_daymin_month_min"),
        pl.col("temperatureMin").alias("air_temp_daymin_month_mean"),
        pl.col("temperatureAvg").alias("air_temp_daymean_month_mean"),
        pl.col("rainAccumulationMax").alias("precipitation_daymax_month_max"),
        pl.col("windSpeedMax").alias("wind_speed_daymax_month_max"),
        pl.col("windSpeedAvg").alias("wind_speed_month_mean"),
        pl.col("cloudCoverAvg").alias("sky_cov"),
        date = pl.lit(datetime.fromisoformat(r_data["timelines"]["daily"][0]["time"])).cast(pl.Date),
        sunshine_duration = train_data["sunshine_duration"].mean(),
        precipitation_month_sum = train_data["precipitation_month_sum"].mean(),
    )
)

inference_data.write_json("data/inference_data.json")