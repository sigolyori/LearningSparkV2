# Databricks notebook source
# MAGIC %fs ls /databricks-datasets/learning-spark-v2/sf-fire/sf-fire-calls.csv

# COMMAND ----------

from pyspark.sql import Row

# COMMAND ----------

row = Row(350, True, 'LearningSparkV2', None)

# COMMAND ----------

row[0]

# COMMAND ----------

row[1]

# COMMAND ----------

row[2]

# COMMAND ----------

