-- Databricks notebook source
-- MAGIC %python
-- MAGIC import re
-- MAGIC username = spark.sql("SELECT current_user()").first()[0]
-- MAGIC clean_username = re.sub("[^a-zA-Z0-9]", "_", username)
-- MAGIC database = f"dbacademy_{clean_username}_adewd_jobs"
-- MAGIC spark.conf.set("da.user_db", f"dbacademy_{clean_username}_adewd_jobs")

-- COMMAND ----------

USE ${da.user_db}

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # ANSWER
-- MAGIC key = "Task 4"

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS task_4
(key STRING, run_time TIMESTAMP);

-- COMMAND ----------

-- MAGIC %python
-- MAGIC spark.sql(f"INSERT INTO task_4 VALUES ('From {key}', current_timestamp())")

-- COMMAND ----------

SELECT COUNT(*) FROM task_4

-- COMMAND ----------

SELECT * FROM task_4
