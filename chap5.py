# Databricks notebook source
from pyspark.sql.types import LongType

# COMMAND ----------

def cubed(s):
    return s * s * s

# COMMAND ----------

spark.range(1, 9).createOrReplaceTempView('udf_test')

# COMMAND ----------

spark.udf.register('cubed', cubed, LongType())

# COMMAND ----------

spark.sql(
"""
SELECT id, cubed(id) AS id_cubed
FROM udf_test
""").show()

# COMMAND ----------

import pandas as pd

from pyspark.sql.functions import col, pandas_udf

from pyspark.sql.types import LongType

# COMMAND ----------

def cubed(a: pd.Series) -> pd.Series:
    return a * a * a

# COMMAND ----------

cubed_udf = pandas_udf(cubed, returnType=LongType())

# COMMAND ----------

df = spark.range(1, 4)

df.select('id', cubed_udf(col('id'))).show()

# COMMAND ----------

x = pd.Series([1,2,3])

print(cubed(x))

# COMMAND ----------

print(cubed_udf(x))

# COMMAND ----------

from pyspark.sql.types import *

# COMMAND ----------

schema = StructType([StructField('celsius', ArrayType(IntegerType()))])

# COMMAND ----------

t_list = [[35,36,32,30,40,42,38]], [[31,32,34,55,56]]

# COMMAND ----------

t_c = spark.createDataFrame(t_list, schema)

# COMMAND ----------

t_c.createOrReplaceTempView('tC')

# COMMAND ----------

t_c.show()

# COMMAND ----------

spark.sql(
"""
SELECT celsius, transform(celsius, t -> ((t * 9 div 5 + 32))) AS fahrenheit
FROM tC
"""
).show()

# COMMAND ----------

spark.sql(
"""
SELECT celsius, filter(celsius, t -> t > 38) AS over_38
FROM tC
"""
).show()

# COMMAND ----------

spark.sql(
"""
SELECT celsius, exists(celsius, t -> t = 38) AS threshold
FROM tC
"""
).show()

# COMMAND ----------

spark.sql(
"""
SELECT celsius,
       reduce(
           celsius,
           0,
           (t, acc) -> t + acc,
           acc -> (acc div size(celsius) * 9 div 5)
              ) AS avgFahrenheit
FROM tC
"""
).show()

# COMMAND ----------

from pyspark.sql.functions import expr

# COMMAND ----------

tripdelaysFilePath = '/databricks-datasets/learning-spark-v2/flights/departuredelays.csv'
airportsnaFilePath = '/databricks-datasets/learning-spark-v2/flights/airport-codes-na.txt'

# COMMAND ----------

airportsna = (spark.read.format('csv')
    .options(header='true', inferSchema='true', sep='\t')
     .load(airportsnaFilePath)
)

# COMMAND ----------

airportsna.createOrReplaceTempView('airports_na')

# COMMAND ----------

departureDelays = (spark.read.format('csv')
    .options(header='true')
     .load(tripdelaysFilePath)
)

# COMMAND ----------

departureDelays = (departureDelays
    .withColumn('delay', expr('CAST(delay as INT) AS delay'))
    .withColumn('ddistance', expr('CAST(delay as INT) AS distance'))
)

# COMMAND ----------

departureDelays.createOrReplaceTempView('departureDelays')

# COMMAND ----------

foo = (departureDelays
    .filter(expr("""origin == 'SEA' AND destination =='SFO' AND date like '01010%' AND delay > 0""")))

# COMMAND ----------

foo.createOrReplaceTempView('foo')

# COMMAND ----------

spark.sql(
"""
SELECT *
FROM airports_na LIMIT 10
"""
).show()

# COMMAND ----------

spark.sql(
"""
SELECT *
FROM departureDelays LIMIT 10
"""
).show()

# COMMAND ----------

spark.sql(
"""
SELECT *
FROM foo
"""
).show()

# COMMAND ----------

(foo.join(
    airportsna,
    airportsna.IATA == foo.origin)
     .select('City', 'State', 'date', 'delay', 'distance', 'destination')
     .show()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT a.City, a.State, f.date, f.delay, f.distance, f.destination
# MAGIC FROM foo f
# MAGIC JOIN airports_na a
# MAGIC ON a.IATA = f.origin

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS departureDelaysWindow;
# MAGIC 
# MAGIC CREATE TABLE departureDelaysWindow AS
# MAGIC SELECT origin, destination, SUM(delay) AS TotalDelays
# MAGIC FROM departureDelays
# MAGIC WHERE origin IN ('SEA', 'SFO', 'JFK')
# MAGIC   AND destination IN ('SEA', 'SFO', 'JFK', 'DEN', 'ORD', 'LAX', 'ATL')
# MAGIC GROUP BY origin, destination

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM departureDelaysWindow

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT origin, destination, SUM(TotalDelays) AS TotalDelays
# MAGIC FROM departureDelaysWindow
# MAGIC WHERE origin = 'JFK'
# MAGIC GROUP BY origin, destination
# MAGIC ORDER BY SUM(TotalDelays) DESC
# MAGIC LIMIT 3

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT origin, destination, TotalDelays,
# MAGIC          dense_rank() OVER (PARTITION BY origin ORDER BY TotalDelays DESC) AS rank
# MAGIC          FROM departureDelaysWindow

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT origin, destination, TotalDelays, rank
# MAGIC FROM (
# MAGIC   SELECT origin, destination, TotalDelays,
# MAGIC          dense_rank() OVER (PARTITION BY origin ORDER BY TotalDelays DESC) AS rank
# MAGIC          FROM departureDelaysWindow
# MAGIC ) t
# MAGIC where rank <= 3

# COMMAND ----------

foo.show()

# COMMAND ----------

from pyspark.sql.functions import expr

foo2 = (foo.withColumn(
            'status',
            expr("""CASE WHEN delay <= 10 THEN 'On-time' ELSE 'Delayed' END"""))
       )

# COMMAND ----------

foo2.show()

# COMMAND ----------

foo3 = foo2.drop('delay')
foo3.show()

# COMMAND ----------

foo4 = foo3.withColumnRenamed('status', 'flight_status')
foo4.show()

# COMMAND ----------

# MAGIC %sql 
# MAGIC 
# MAGIC select destination, cast(substring(date, 0, 2) as int) as month, delay
# MAGIC from departureDelays
# MAGIC where origin = 'SEA'

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * 
# MAGIC from (
# MAGIC   select destination, cast(substring(date, 0, 2) as int) as month, delay
# MAGIC   from departureDelays
# MAGIC   where origin = 'SEA')
# MAGIC pivot (
# MAGIC   cast(avg(delay) as decimal(4, 2)) as AvgDelay, max(delay) as MaxDelay
# MAGIC   for month in (1 JAN, 2 FEB, 3 MAR)
# MAGIC )
# MAGIC order by destination