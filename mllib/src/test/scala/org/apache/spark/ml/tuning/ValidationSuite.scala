package org.apache.spark.ml.tuning

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasInputCol
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType

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