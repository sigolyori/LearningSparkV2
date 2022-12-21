# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = (SparkSession
        .builder
        .appName('SparkSQLExampleApp')
        .getOrCreate())

# COMMAND ----------

csv_file = '/databricks-datasets/learning-spark-v2/flights/departuredelays.csv'

# COMMAND ----------

df = (spark.read.format('csv')
     .option('inferSchema', 'true')
     .option('header', 'true')
     .load(csv_file))

# COMMAND ----------

df.createOrReplaceTempView('us_delay_flights_tbl')

# COMMAND ----------

spark.sql(
"""
SELECT *
FROM us_delay_flights_tbl
LIMIT 100
""").show()

# COMMAND ----------

spark.sql(
    """
    SELECT distance, origin, destination 
    FROM us_delay_flights_tbl
    WHERE distance > 1000
    ORDER BY distance DESC
    """).show(10)

# COMMAND ----------

spark.sql(
"""
select 
    date, delay, origin, destination
from
    us_delay_flights_tbl
where
    delay > 120 and origin = 'SFO' and destination = 'ORD'
order by
    delay desc
""").show(10)

# COMMAND ----------

spark.sql(
"""
select 
    to_date(cast((date, 'MMddhhmm'))), delay
from
    us_delay_flights_tbl
""").show(10)

# COMMAND ----------

spark.sql(
'''
select to_date(cast(date as string), 'MMddHHmm')
from us_delay_flights_tbl
'''
).show()

# COMMAND ----------

spark.sql(
"""
select *, date, to_date_format_udf(date) as date_fm 
from us_delay_flights_tbl
""").show(10)

# COMMAND ----------

spark.sql(
"""
SELECT delay, origin, destination,
CASE 
    WHEN delay > 360 THEN 'Very Long'
    WHEN delay >= 120 AND delay <= 360 THEN 'Long Delays'
    WHEN delay >= 60 AND delay <= 120 THEN 'Short Delays'
    WHEN delay >= 0 AND delay <= 60 THEN 'Tolerable Delays'
    WHEN delay = 0 THEN 'No Delays'
    ELSE 'Early'
END AS Flight_Delays
FROM us_delay_flights_tbl
ORDER BY origin, delay DESC
"""
).show(10)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# spark.sql(
# """
# select 
#     date, delay, origin, destination
# from
#     us_delay_flights_tbl
# where
#     delay > 120 and origin = 'SFO' and destination = 'ORD'
# order by
#     delay desc
# """).show(10)



# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

(df
    .select('date', 'delay', 'origin', 'destination')
    .where((col('delay') > 120) & (col('origin') == 'SFO') & (col('destination') == 'ORD'))
    .orderBy('delay')
    .show(10)
)

# COMMAND ----------

# spark.sql(
# """
# SELECT delay, origin, destination,
# CASE 
#     WHEN delay > 360 THEN 'Very Long'
#     WHEN delay >= 120 AND delay <= 360 THEN 'Long Delays'
#     WHEN delay >= 60 AND delay <= 120 THEN 'Short Delays'
#     WHEN delay >= 0 AND delay <= 60 THEN 'Tolerable Delays'
#     WHEN delay = 0 THEN 'No Delays'
#     ELSE 'Early'
# END AS Flight_Delays
# FROM us_delay_flights_tbl
# ORDER BY origin, delay DESC
# """
# ).show(10)

# COMMAND ----------

(df
    .select('delay', 'origin', 'destination')
    .withColumn('delay', when((df.delay > 360), 'Very Long Delays')
                        .when((df.delay >= 120) & (df.delay <= 360), 'Long Delays')
                        .when((df.delay >= 60) & (df.delay <= 120), 'Short Delays')
                        .when((df.delay >= 60) & (df.delay <= 120), 'Tolerable Delays')
                        .when(df.delay == 0, 'No Delays')
                        .otherwise('Early'))
    .show()
)

# COMMAND ----------

(df
    .select('delay', 'origin', 'destination')
    .withColumn('delay', 
                expr("""
                CASE 
                WHEN delay > 360 THEN 'Very Long'
                WHEN delay >= 120 AND delay <= 360 THEN 'Long Delays'
                WHEN delay >= 60 AND delay <= 120 THEN 'Short Delays'
                WHEN delay >= 0 AND delay <= 60 THEN 'Tolerable Delays'
                WHEN delay = 0 THEN 'No Delays'
                ELSE 'Early'
                """)
                )
)

# COMMAND ----------

spark.sql('CREATE DATABASE learn_spark_db')

# COMMAND ----------

bspark.sql('USE learn_spark_db')

# COMMAND ----------

spark.sql(
"""
CREATE TABLE managed_us_delay_flights_tbl (date STRING, delay INT, distance INT, origin STRING, destination STRING)
"""
)

# COMMAND ----------

df = (
    spark.read.format('csv')
    .schema('date STRING, delay INT, distance INT, origin STRING, destination STRING')
    .option('header', 'true')
    .option('path', "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv")
    .load()
)

# COMMAND ----------

df.write.mode('overwrite').saveAsTable('us_delay_flights_tbl')

# COMMAND ----------

spark.sql(
"""
CREATE OR REPLACE GLOBAL TEMP VIEW us_origin_airport_SFO_global_tmp_view AS
    SELECT date, delay, origin, destination
    FROM us_delay_flights_tbl
    WHERE origin = 'SFO'
""")

# COMMAND ----------

spark.sql(
"""
CREATE OR REPLACE TEMP VIEW us_origin_airport_JFK_tmp_view AS
    SELECT date, delay, origin, destination
    FROM us_delay_flights_tbl
    WHERE origin = 'JFK'
""")


# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM global_temp.us_origin_airport_SFO_global_tmp_view

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM us_origin_airport_JFK_tmp_view

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS us_origin_airport_JFK_tmp_view;
# MAGIC DROP VIEW IF EXISTS us_origin_airport_SFO_global_tmp_view;

# COMMAND ----------

spark.catalog.listDatabases()

# COMMAND ----------

spark.catalog.listTables()

# COMMAND ----------

spark.catalog.listColumns('us_delay_flights_tbl')

# COMMAND ----------

# MAGIC %sql
# MAGIC CACHE LAZY TABLE us_delay_flights_tbl

# COMMAND ----------

# MAGIC %sql
# MAGIC UNCACHE TABLE us_delay_flights_tbl

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM us_delay_flights_tbl

# COMMAND ----------

us_flights_df = spark.sql("""SELECT * FROM us_delay_flights_tbl""")

# COMMAND ----------

us_flights_df2 = spark.table('us_delay_flights_tbl')

# COMMAND ----------

type(us_flights_df) == type(us_flights_df2)

# COMMAND ----------

file = '''/databricks-datasets/learning-spark-v2/flights/summary-data/parquet/2010-summary.parquet'''

# COMMAND ----------

df = spark.read.format('parquet').load(file)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW us_flights_tbl
# MAGIC   USING parquet
# MAGIC   OPTIONS (
# MAGIC     path "/databricks-datasets/learning-spark-v2/flights/summary-data/parquet/2010-summary.parquet")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM us_flights_tbl

# COMMAND ----------

file = """/databricks-datasets/learning-spark-v2/flights/summary-data/parquet/2010-summary.parquet"""
df = spark.read.format('parquet').load(file)

# COMMAND ----------

(df.write.format('parquet')
    .mode('overwrite')
    .option('compression', 'snappy')
    .save('tmp/data/parquet/df_parquet'))

# COMMAND ----------

(df.write
    .mode('overwrite')
    .saveAsTable('us_delay_flights_tbl')
)

# COMMAND ----------

file = """/databricks-datasets/learning-spark-v2/flights/summary-data/json/*"""
df = spark.read.format('json').load(file)

# COMMAND ----------

file = """/databricks-datasets/learning-spark-v2/flights/summary-data/csv/*"""
schema = 'DEST_COUNTRY_NAME STRING, ORIGIN_COUNTRY_NAME STRING, count INT'

# COMMAND ----------

df = (spark.read.format('csv')
          .option('header', 'true')
          .schema(schema)
          .option('mode', 'FAILFAST')
          .option('nullValue', '')
          .load(file)
     )

# COMMAND ----------

df.show()

# COMMAND ----------

from pyspark.ml import image

# COMMAND ----------

image_dir = """/databricks-datasets/learning-spark-v2/cctvVideos/train_images/"""
images_df = spark.read.format('image').load(image_dir)
images_df.printSchema()

# COMMAND ----------

images_df.select('image.height', 'image.width', 'image.nChannels', 'image.mode', 'label').show(10, truncate = False)

# COMMAND ----------

images_df.show(5)

# COMMAND ----------

  path = """/databricks-datasets/learning-spark-v2/cctvVideos/train_images/"""

# COMMAND ----------

binary_files_df = (spark.read.format('binaryFile')
    .option('pathGlobFilter', '*.jpg')
    .load(path))
binary_files_df.show(5)

# COMMAND ----------

