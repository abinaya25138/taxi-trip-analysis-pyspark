# Taxi Trip Analysis with PySpark

A big-data-style analysis of ride data using **PySpark**, built around a ride-hailing business problem (demand patterns, trip economics, pickup hotspots) on the publicly available **NYC TLC taxi trip dataset** as a stand-in for proprietary ride-hailing data.

## Pipeline

1. **Ingestion** — load ~100K trip records into a Spark DataFrame (`nyc_taxi_trips.csv`).
2. **Dimensional modeling** — extract `datetime_dim`, `pickup_location_dim`, and `dropoff_location_dim` tables (a small star-schema pattern for the trip fact table).
3. **Data quality checks** — count missing values per column; summary statistics via `.describe()`.
4. **Feature engineering** — derive `trip_distance_in_miles`, pickup/dropoff hour & minute, `trip_duration_minutes`, `trip_average_speed_mph`, and a derived `estimated_tip_amount` label.
5. **Predictive modeling** — a Spark MLlib `LinearRegression` estimates tip amount from trip features; evaluated with RMSE and R² on a held-out test split.
6. **Clustering** — `KMeans` groups pickup coordinates into 5 hotspot zones, visualized with a scatter plot.

## Tech stack

PySpark (DataFrame API + MLlib), pandas, matplotlib, seaborn

## Setup

```bash
pip install pyspark pandas matplotlib seaborn
jupyter notebook ola_bigdata.ipynb
```

Requires a working Java runtime (PySpark needs a JVM — Java 8/11/17 all work).

## Notes

- The dataset models a ride-hailing scenario but is the public NYC taxi trip dataset, not proprietary Ola data — real Ola data isn't publicly available, so this stands in for it.
- `estimated_tip_amount` is a simplified label (20% of fare) used to demonstrate the regression pipeline, not a real-world tip model.

## Possible extensions

- Swap in a tree-based model (Random Forest / GBT) for the tip-prediction task and compare RMSE
- Use the pickup cluster id as a categorical feature in the regression
- Add time-series demand forecasting by hour/day-of-week
