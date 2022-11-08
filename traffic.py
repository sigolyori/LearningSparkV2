# Databricks notebook source
path = 'gs://hyk/train.csv'

# COMMAND ----------

import pandas as pd

# COMMAND ----------

pd.read_csv(path, nrows=100)

# COMMAND ----------

# schema('id string, base_date int, day_of_week string, base_hour int, lane_count int, road_rating int, road_name string, multi_linked int, connect_code int, maximum_speed_limit float, vehicle_restricted float, weight_restricted float, height_restricted float, road_type string, start_node_name string, start_latitude float, start_longtitude float, end_latitude float, end_longtitude float, start_turn_restricted string, end_node_name string, end_turn_restricted string, target float').load()

# COMMAND ----------

path

# COMMAND ----------

df = (spark.read.format('csv')
    .option('header', 'true')
    .option('path', path)
    .option('inferSchema', 'true')
    .load()  
     )

# COMMAND ----------

(df
    .show())