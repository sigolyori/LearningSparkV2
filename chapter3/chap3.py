# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

# COMMAND ----------

spark = (SparkSession
        .builder
        .appName('AuthorsAges')
        .getOrCreate())

# COMMAND ----------

data_df = spark.createDataFrame([('Brooke', 20), ('Denny', 31), ('Jules', 30), ('TD', 35), ('Brooke', 25)], ['name', 'age'])

# COMMAND ----------

avg_df = data_df.groupBy('name').agg(avg('age'))

# COMMAND ----------

avg_df.show()

# COMMAND ----------

from pyspark.sql.types import *

# COMMAND ----------

schema = StructType([StructField('author', StringType(), False),
           StructField('title', StringType(), False),
           StructField('pages', IntegerType(), False)])

# COMMAND ----------

schema = 'author STRING, title STRING, pages INT'

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

schema =  "'Id' INT, 'First' STRING, 'Last' STRING, 'Url' STRING, 'Published' STRING, 'Hits' INT, 'Campaigns' ARRAY<STRING>"

# COMMAND ----------

data = [[1, "Jules", "Damji", "https://tinyurl.1", "1/4/2016", 4535, ["twitter", "LinkedIn"]],
       [2, "Brooke","Wenig","https://tinyurl.2", "5/5/2018", 8908, ["twitter", "LinkedIn"]],
       [3, "Denny", "Lee", "https://tinyurl.3","6/7/2019",7659, ["web", "twitter", "FB", "LinkedIn"]],
       [4, "Tathagata", "Das","https://tinyurl.4", "5/12/2018", 10568, ["twitter", "FB"]],
       [5, "Matei","Zaharia", "https://tinyurl.5", "5/14/2014", 40578, ["web", "twitter", "FB", "LinkedIn"]],
       [6, "Reynold", "Xin", "https://tinyurl.6", "3/2/2015", 25568, ["twitter", "LinkedIn"]]
       ]

# COMMAND ----------

from pyspark.sql import Row

# COMMAND ----------

blog_row = Row(6, 'Reynold', 'Xin', 'https://tinyurl.6', 255568, '3/2/2015')

# COMMAND ----------

blog_row[1]

# COMMAND ----------

rows = [Row('Heeyoung Kim', 'KR'), Row('Reynold Xin', 'CA')]

# COMMAND ----------

authors_df = spark.createDataFrame(rows, schema = ['Authors', 'State'])
authors_df.show()

# COMMAND ----------

