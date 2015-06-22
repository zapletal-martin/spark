package org.apache.spark.ml.tuning

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType

/**
 * Params for [[CrossValidator]] and [[CrossValidatorModel]].
 */
private[ml] trait ValidationParams extends Params {

  /**
   * param for the estimator to be cross-validated
   * @group param
   */
  val estimator: Param[Estimator[_]] = new Param(this, "estimator", "estimator for selection")

  /** @group getParam */
  def getEstimator: Estimator[_] = $(estimator)

  /**
   * param for estimator param maps
   * @group param
   */
  val estimatorParamMaps: Param[Array[ParamMap]] =
    new Param(this, "estimatorParamMaps", "param maps for the estimator")

  /** @group getParam */
  def getEstimatorParamMaps: Array[ParamMap] = $(estimatorParamMaps)

  /**
   * param for the evaluator used to select hyper-parameters that maximize the cross-validated
   * metric
   * @group param
   */
  val evaluator: Param[Evaluator] = new Param(this, "evaluator",
    "evaluator used to select hyper-parameters that maximize the cross-validated metric")

  /** @group getParam */
  def getEvaluator: Evaluator = $(evaluator)
}

@Experimental
trait Validation[M <: Model[M]]
  extends Estimator[M]
  with Logging
  with ValidationParams {

  /** @group setParam */
  def setEstimator(value: Estimator[_]): this.type = set(estimator, value)

  /** @group setParam */
  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  /** @group setParam */
  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  override def transformSchema(schema: StructType): StructType = {
    $(estimator).transformSchema(schema)
  }

  override def validateParams(): Unit = {
    super.validateParams()
    val est = $(estimator)
    for (paramMap <- $(estimatorParamMaps)) {
      est.copy(paramMap).validateParams()
    }
  }
}

/**
 * :: Experimental ::
 * Model from k-fold cross validation.
 */
@Experimental
abstract class ValidationModel[M <: Model[M]] private[ml] (
    override val uid: String,
    val bestModel: Model[_],
    val avgMetrics: Array[Double])
  extends Model[M] {

  override def validateParams(): Unit = {
    bestModel.validateParams()
  }

  override def transform(dataset: DataFrame): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    bestModel.transform(dataset)
  }

  override def transformSchema(schema: StructType): StructType = {
    bestModel.transformSchema(schema)
  }
}
