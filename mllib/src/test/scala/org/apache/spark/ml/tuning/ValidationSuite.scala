package org.apache.spark.ml.tuning

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.classification.LogisticRegressionSuite._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator, RegressionEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasInputCol
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.classification.LogisticRegressionSuite.generateLogisticInput
import org.apache.spark.mllib.util.{LinearDataGenerator, MLlibTestSparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.types.StructType

class ValidationSuite extends SparkFunSuite with MLlibTestSparkContext {
  @transient var dataset: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val sqlContext = new SQLContext(sc)
    dataset = sqlContext.createDataFrame(
      sc.parallelize(generateLogisticInput(1.0, 1.0, 100, 42), 2))
  }
}

object ValidationSuite {

  abstract class MyModel extends Model[MyModel]

  class MyEstimator(override val uid: String) extends Estimator[MyModel] with HasInputCol {

    override def validateParams(): Unit = require($(inputCol).nonEmpty)

    override def fit(dataset: DataFrame): MyModel = {
      throw new UnsupportedOperationException
    }

    override def transformSchema(schema: StructType): StructType = {
      throw new UnsupportedOperationException
    }

    override def copy(extra: ParamMap): MyEstimator = defaultCopy(extra)
  }

  class MyEvaluator extends Evaluator {

    override def evaluate(dataset: DataFrame): Double = {
      throw new UnsupportedOperationException
    }

    override val uid: String = "eval"

    override def copy(extra: ParamMap): MyEvaluator = defaultCopy(extra)
  }
}