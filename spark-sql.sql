-- Databricks notebook source
CREATE TABLE people (name STRING, age INT);

-- COMMAND ----------

INSERT INTO people VALUES ('Michael', NULL);
INSERT INTO people VALUES ('Heeyoung', 25);
INSERT INTO people VALUES ('Yurim', 25);

-- COMMAND ----------

SELECT * FROM people;

-- COMMAND ----------

SHOW TABLES;

-- COMMAND ----------

SELECT * 
FROM people 
WHERE age < 20;

-- COMMAND ----------

SELECT *
FROM people
WHERE age IS NULL;

-- COMMAND ----------

