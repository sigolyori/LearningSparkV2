-- Databricks notebook source
-- MAGIC %md
-- MAGIC # ![](https://redislabs.com/wp-content/uploads/2016/12/lgo-partners-databricks-125x125.png)  Delta Lake Quickstart  
-- MAGIC <br>
-- MAGIC 이 노트북에서는 SparkSQL 을 사용해서 아래와 같이 Delta Lake 형식의 데이터를 다루는 다양한 방법에 대해서 다룹니다.  
-- MAGIC 
-- MAGIC 
-- MAGIC * Delta Table을 만들고 다양한 DML문들을 사용해서 데이터를 수정하고 정제
-- MAGIC * Delta Table의 구조 이해
-- MAGIC * Time Travel 을 이용한 Table History 관리   
-- MAGIC * 댜양한 IO 최적화 기능 

-- COMMAND ----------

-- MAGIC %md

-- COMMAND ----------

-- DBTITLE 1,setup
-- MAGIC %python
-- MAGIC databricks_user = spark.sql("SELECT current_user()").collect()[0][0].split('@')[0].replace(".", "_")
-- MAGIC print(databricks_user)
-- MAGIC 
-- MAGIC spark.sql("DROP DATABASE IF EXISTS delta_{}_db CASCADE".format(str(databricks_user)))
-- MAGIC spark.sql("CREATE DATABASE IF NOT EXISTS delta_{}_db".format(str(databricks_user)))
-- MAGIC spark.sql("USE delta_{}_db".format(str(databricks_user)))

-- COMMAND ----------

-- DBTITLE 1,Delta Table의 생성
CREATE TABLE IF NOT EXISTS students 
  (id INT, name STRING, value DOUBLE);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Full DML Support
-- MAGIC <br>
-- MAGIC 일반적인 Data Lake의 경우 모든 데이터는 Append Only임을 가정합니다.  
-- MAGIC 
-- MAGIC Delta Lake를 사용하면 마치 Database를 사용하는 것처럼 Insert,Update,Delete를 사용해서 손쉽게 데이터셋을 수정할 수 있습니다. 

-- COMMAND ----------

INSERT INTO students VALUES (1, "Yve", 1.0);
INSERT INTO students VALUES (2, "Omar", 2.5);
INSERT INTO students VALUES (3, "Elia", 3.3);

-- COMMAND ----------

INSERT INTO students
VALUES 
  (4, "Ted", 4.7),
  (5, "Tiffany", 5.5),
  (6, "Vini", 6.3);
  

-- COMMAND ----------

UPDATE students 
SET value = value + 1
WHERE name LIKE "T%"

-- COMMAND ----------

DELETE FROM students 
WHERE value > 6

-- COMMAND ----------

SELECT * FROM students;

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ### ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) MERGE 를 사용한 Upsert 수행 
-- MAGIC 
-- MAGIC Databricks에서는 MERGE문을 사용해서 Upsert- 데이터의 Update,Insert 및 기타 데이터 조작을 하나의 명령어로 수행합니다.  
-- MAGIC 아래의 예제는 변경사항을 기록하는 CDC(Change Data Capture) 로그데이터를 updates라는 임시뷰로 생성합니다. 

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW updates(id, name, value, type) AS VALUES
  (2, "Omar", 15.2, "update"),
  (3, "", null, "delete"),
  (7, "Blue", 7.7, "insert"),
  (11, "Diya", 8.8, "update");
  
SELECT * FROM updates;

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC 이 view에는 레코드들에 대한 3가지 타입- insert,update,delete 명령어 기록을 담고 있습니다.  
-- MAGIC 이 명령어를 각각 수행한다면 3개의 트렌젝션이 되고 만일 이중에 하나라도 실패하게 된다면 invalid한 상태가 될 수 있습니다.  
-- MAGIC 대신에 이 3가지 action을 하나의 atomic 트렌젝션으로 묶어서 한꺼번에 적용되도록 합니다.  
-- MAGIC <br>
-- MAGIC **`MERGE`**  문은 최소한 한 하나의 기준 field (여기서는 id)를 가지고 각 **`WHEN MATCHED`** 이나 **`WHEN NOT MATCHED`**  구절은 여러 조건값들을 가질 수 있습니다.  
-- MAGIC **id** 필드를 기준으로 **type** 필드값에 따라서 각 record에 대해서 update,delete,insert문을 수행하게 됩니다. 

-- COMMAND ----------

MERGE INTO students b
USING updates u
ON b.id=u.id
WHEN MATCHED AND u.type = "update"
  THEN UPDATE SET *
WHEN MATCHED AND u.type = "delete"
  THEN DELETE
WHEN NOT MATCHED AND u.type = "insert"
  THEN INSERT *

-- COMMAND ----------

SELECT * FROM students;

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Inside Delta 
-- MAGIC Delta Lake를 이루는 테이블의 내부 구조를 알아보자

-- COMMAND ----------

-- DBTITLE 1,Location 행에서 테이블을 이루는 파일의 위치 정보를 확인합니다
DESCRIBE EXTENDED students

-- COMMAND ----------

-- MAGIC % md
-- MAGIC 테이블이 아니라 데이터레이크!

-- COMMAND ----------

-- DBTITLE 1,Delta Lake File 을 조사해 보자.
-- MAGIC %python
-- MAGIC display(dbutils.fs.ls(f"/user/hive/warehouse/delta_{databricks_user}_db.db/students"))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(dbutils.fs.ls(f"/user/hive/warehouse/delta_{databricks_user}_db.db/students/_delta_log"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC 데이터를 실제로 지우는 것이 아니라, 버전을 가지고 있음

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(spark.sql(f"SELECT * FROM json.`dbfs:/user/hive/warehouse/delta_{databricks_user}_db.db/students/_delta_log/00000000000000000007.json`"))

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Table History를 사용한 Time Travel 기능 

-- COMMAND ----------

DESCRIBE HISTORY students

-- COMMAND ----------

-- DBTITLE 1,과거 버전의 데이터 조회
SELECT * FROM students VERSION AS OF 2;
-- SELECT * FROM students@v2;

-- COMMAND ----------

-- DBTITLE 1,과거 버전으로 돌아가기
--RESTORE TABLE students TO VERSION AS OF 2;
select * from students@v2

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Delta Data file에 대한 최적화 
-- MAGIC 
-- MAGIC 이런저런 작업을 하다 보면 필연적으로 굉장히 작은 데이터 파일들이 많이 생성되게 됩니다.  
-- MAGIC 성능 향상을 위해서 이런 파일들에 대한 최적화하는 방법과 불필요한 파일들을 정리하는 명령어들에 대해서 알아봅시다. 

-- COMMAND ----------

DESCRIBE DETAIL students

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **`OPTIMIZE`** 명령어는 기존의 데이터 파일내의 레코드들을 합쳐서 새로 최적의 사이즈로 파일을 만들고 기존의 작은 파일들을 읽기 성능이 좋은 큰 파일들로 대체합니다.  
-- MAGIC 이 떄 옵션값으로 하나 이상의 필드를 지정해서 **`ZORDER`** 인덱싱을 수행할 수 있습니다.  
-- MAGIC Z-Ordering은 관련 정보를 동일한 파일 집합에 배치해서 읽어야 하는 데이터의 양을 줄여 쿼리 성능을 향상 시키는 기술입니다. 쿼리 조건에 자주 사용되고 해당 열에 높은 카디널리티(distinct 값이 많은)가 있는 경우 `ZORDER BY`를 사용합니다.

-- COMMAND ----------

OPTIMIZE students ZORDER BY id

-- COMMAND ----------

DESCRIBE HISTORY students

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Stale File 정리하기
-- MAGIC Databricks는 자동으로 Delta Lake Table 에서 불필요한 파일들을 정리합니다.  
-- MAGIC Delta Lake의 Versioning과 Time Travel은 과거 버전을 조회하고 실수했을 경우 데이터를 rollback하는 매우 유용한 기능이지만, 데이터 파일의 모든 버전을 영구적으로 저장하는 것은 비용이 많이 들게 됩니다.  
-- MAGIC 기본값으로 **VACUUM** 은 7일 미만의 데이터를 삭제하지 못하도록 합니다. 아래의 예제는 이 기본 설정을 무시하고 가장 최근 버전 데이터만 남기고 모든 과거 버전의 stale file을 정리하는 예제입니다. 

-- COMMAND ----------

SET spark.databricks.delta.retentionDurationCheck.enabled = false;
SET spark.databricks.delta.vacuum.logging.enabled = true;

VACUUM students RETAIN 0 HOURS

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(dbutils.fs.ls(f"/user/hive/warehouse/students"))

-- COMMAND ----------

-- DBTITLE 1,Table 정리 
drop table students;

-- COMMAND ----------

select * from students;

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>

-- COMMAND ----------

