package org.apache.spark.ml.tuning

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import scala.reflect.runtime.universe._

import scala.reflect.ClassTag

/**
 * Common params for [[TrainValidationSplitParams]] and [[CrossValidatorParams]].
 */
private[ml] trait ValidationParams extends Params {

  /**
   * param for the estimator to be validated
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
   * param for the evaluator used to select hyper-parameters that maximize the validated metric
   * @group param
   */
  val evaluator: Param[Evaluator] = new Param(this, "evaluator",
    "evaluator used to select hyper-parameters that maximize the validated metric")

  /** @group getParam */
  def getEvaluator: Evaluator = $(evaluator)
}

@Experimental
private[ml] abstract class Validation[M <: Model[M], V <: Validation[M, _] : ClassTag]
  (override val uid: String)
  extends Estimator[M]
  with Logging with ValidationParams {

  def this() = this(Identifiable.randomUID("cv"))

  private val f2jBLAS = new F2jBLAS

  /** @group setParam */
  def setEstimator(value: Estimator[_]): V = set(estimator, value).asInstanceOf[V]

  /** @group setParam */
  def setEstimatorParamMaps(value: Array[ParamMap]): V = set(estimatorParamMaps, value).asInstanceOf[V]

  /** @group setParam */
  def setEvaluator(value: Evaluator): V = set(evaluator, value).asInstanceOf[V]

  protected def measureModels(
      trainingDataset: DataFrame,
      validationDataset: DataFrame,
      est: Estimator[_],
      eval: Evaluator,
      epm: Array[ParamMap],
      numModels: Int) = {

    val metrics = new Array[Double](numModels)
    val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
    trainingDataset.unpersist()
    var i = 0

    // multi-model training
    while (i < numModels) {
      // TODO: duplicate evaluator to take extra params from input
      val metric = eval.evaluate(models(i).transform(validationDataset, epm(i)))
      logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
      metrics(i) += metric
      i += 1
    }
    validationDataset.unpersist()

    metrics
  }
/*
  def validate(
      dataset: DataFrame,
      model: (String, Model[_], Array[Double]) => M,
      validationMetrics: DataFrame => Array[Double]) = {

    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sqlCtx = dataset.sqlContext
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length

    val metrics = new Array[Double](epm.length)
    val splits = MLUtils.sample(dataset.rdd, 2, 3d/4d, 4d/4d)
    val training = splits._1
    val validation = splits._2

    val trainingDataset = sqlCtx.createDataFrame(training, schema).cache()
    val validationDataset = sqlCtx.createDataFrame(validation, schema).cache()
    // multi-model training
    val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
    trainingDataset.unpersist()
    var i = 0
    while (i < numModels) {
      // TODO: duplicate evaluator to take extra params from input
      val metric = eval.evaluate(models(i).transform(validationDataset, epm(i)))
      logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
      metrics(i) += metric
      i += 1
    }
    validationDataset.unpersist()

    f2jBLAS.dscal(numModels, 1.0, metrics, 1)
    logInfo(s"Average validation metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) = metrics.zipWithIndex.maxBy(_._1)
    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    copyValues(new TrainValidationSplitModel(uid, bestModel, metrics).setParent(this))
  }*/

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

  override def copy(extra: ParamMap): V = {
    val copied = defaultCopy(extra).asInstanceOf[V]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }
}

/**
 * :: Experimental ::
 * Model from validation.
 */
@Experimental
private[ml] abstract class ValidationModel[M <: Model[M]] private[ml] (
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