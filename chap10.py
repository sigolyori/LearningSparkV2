# Databricks notebook source
filePath = """/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"""

# COMMAND ----------

airbnbDF = spark.read.parquet(filePath)

# COMMAND ----------

airbnbDF.select('neighbourhood_cleansed', 'room_type', 'bedrooms', 'bathrooms',
               'number_of_reviews', 'price').show(5)

# COMMAND ----------

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed = 42)
print(f"""There are {trainDF.count()} rows in the training set,
          and {testDF.count()} in the test set""")

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=['bedrooms'], outputCol = 'features')
vecTrainDF = vecAssembler.transform(trainDF)

# COMMAND ----------

vecTrainDF.select('bedrooms', 'features', 'price').show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

lr = LinearRegression(featuresCol = 'features', labelCol = 'price')
lrModel = lr.fit(vecTrainDF)

# COMMAND ----------

m = round(lrModel.coefficients[0], 2)
b = round(lrModel.intercept, 2)
print(f"""The formula for the linear regression line is 
          price = {m} * bedrooms + {b}""")

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages = [vecAssembler, lr])
pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

predDF = pipelineModel.transform(testDF)

# COMMAND ----------

predDF.select('bedrooms', 'features', 'price', 'prediction').show(10)

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

# COMMAND ----------

categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == 'string']
indexOutputCols = [x + 'Index' for x in categoricalCols]
oheOutputCols = [x + 'OHE' for x in categoricalCols]

# COMMAND ----------

stringIndexer = StringIndexer(inputCols = categoricalCols,
                             outputCols = indexOutputCols,
                             handleInvalid = 'skip')

# COMMAND ----------

oheEncoder = OneHotEncoder(inputCols=indexOutputCols,
                          outputCols=oheOutputCols)

# COMMAND ----------

numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == 'double') & (field != 'price'))]

# COMMAND ----------

assemblerInputs = oheOutputCols + numericCols

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols = assemblerInputs, outputCol='features')

# COMMAND ----------

from pyspark.ml.feature import RFormula

# COMMAND ----------

rFormula = RFormula(formula = 'price ~ .',
                   featuresCol = 'features',
                   labelCol = 'price',
                   handleInvalid='skip')

# COMMAND ----------

lr = LinearRegression(labelCol = 'price', featuresCol='features')
pipeline = Pipeline(stages = [stringIndexer, oheEncoder, vecAssembler, lr])
# pipeline = Pipeline(stages = [rFormula, lr])

# COMMAND ----------

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

predDF = pipelineModel.transform(testDF)

# COMMAND ----------

predDF.select('features', 'price', 'prediction').show(5, truncate = False)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
regressionEvaluator = RegressionEvaluator(
    predictionCol = 'prediction',
    labelCol = 'price',
    metricName = 'rmse')
rmse = regressionEvaluator.evaluate(predDF)
print(f'RMSE is {rmse:.1f}')

# COMMAND ----------

r2 = regressionEvaluator.setMetricName('r2').evaluate(predDF)

# COMMAND ----------

print(f'R2 is {r2}')

# COMMAND ----------

pipelinePath = '/tmp/lr-pipeline-model'
pipelineModel.write().overwrite().save(pipelinePath)

# COMMAND ----------

from pyspark.ml import PipelineModel

# COMMAND ----------

savedPipelineModel = PipelineModel.load(pipelinePath)

# COMMAND ----------

savedPipelineModel

# COMMAND ----------

# 의사결정나무 (하이퍼파라미터튜닝)

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

# COMMAND ----------

dt = DecisionTreeRegressor(labelCol = 'price')

# COMMAND ----------

numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == 'double') & (field != 'price'))]

# COMMAND ----------

assemblerInputs = indexOutputCols + numericCols

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols= assemblerInputs, outputCol='features')

# COMMAND ----------

stages = [stringIndexer, vecAssembler, dt]

# COMMAND ----------

pipeline = Pipeline(stages = stages)

# COMMAND ----------

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

dt.setMaxBins(40)

# COMMAND ----------

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

pipelineModel.stages

# COMMAND ----------

dtModel = pipelineModel.stages[-1]

# COMMAND ----------

print(dtModel.toDebugString)

# COMMAND ----------

import pandas as pd

featureImp = pd.DataFrame(
    list(zip(vecAssembler.getInputCols(), dtModel.featureImportances)),
    columns = ['feature', 'importance'])

# COMMAND ----------

featureImp.sort_values(by = 'importance', ascending=False)

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor(labelCol = 'price', maxBins = 40, seed = 42)

# COMMAND ----------

pipeline = Pipeline(stages = [stringIndexer, vecAssembler, rf])

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

# COMMAND ----------

paramGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth, [2,4,6])
            .addGrid(rf.numTrees, [10, 100])
             .build()
            )

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol='price',
                   predictionCol='prediction',
                   metricName='rmse')

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

# COMMAND ----------

cv = CrossValidator(estimator=pipeline,
                    evaluator=evaluator,
                   estimatorParamMaps=paramGrid,
                   numFolds=3,
                   seed=42)

# COMMAND ----------

cvModel = cv.fit(trainDF)

# COMMAND ----------

list(zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics))

# COMMAND ----------

cvModel = cv.setParallelism(4).fit(trainDF)