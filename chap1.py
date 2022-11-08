# Databricks notebook source
log_df = spark.table("default.train_csv").repartition(8)
print(log_df.rdd.getNumPartitions())

# COMMAND ----------

df = spark.range(0, 10000, 1, 8)
print(df.rdd.getNumPartitions())

# COMMAND ----------

