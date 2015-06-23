package org.apache.spark.ml.tuning

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame

/**
 * Params for [[TrainValidationSplit]] and [[TrainValidationSplitModel]].
 */
private[ml] trait TrainValidationSplitParams extends ValidationParams {
  /**
   * Param for ratio between train and validation data. Must be between 0 and 1.
   * Default: 0.75
   * @group param
   */
  val trainRatio: DoubleParam = new DoubleParam(this, "numFolds",
    "ratio between training set and validation set (>= 0 && <= 1)", ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getTrainPercent: Double = $(trainRatio)

  setDefault(trainRatio -> 0.75)
}

/**
 * :: Experimental ::
 * Validation for hyper-parameter tuning.
 * Randomly splits the input dataset into train and validation sets.
 * And uses evaluation metric on the validation set to select the best model.
 * Similar to CrossValidator, but only splits the set once.
 */
@Experimental
class TrainValidationSplit(uid: String)
  extends Validation[TrainValidationSplitModel, TrainValidationSplit](uid)
  with TrainValidationSplitParams with Logging {

  def this() = this(Identifiable.randomUID("cv"))

  /** @group setParam */
  def setTrainRatio(value: Double): this.type = set(trainRatio, value)

  override protected def validationLogic(dataset: DataFrame, est: Estimator[_], eval: Evaluator, epm: Array[ParamMap], numModels: Int): Array[Double] = {
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sqlCtx = dataset.sqlContext

    val splits = MLUtils.sample(dataset.rdd, 2, $(trainRatio), 4d/4d)
    val training = splits._1
    val validation = splits._2

    val trainingDataset = sqlCtx.createDataFrame(training, schema).cache()
    val validationDataset = sqlCtx.createDataFrame(validation, schema).cache()

    val metrics = measureModels(trainingDataset, validationDataset, est, eval, epm, numModels)
    f2jBLAS.dscal(numModels, 1.0, metrics, 1)
    metrics
  }

  override protected def createModel(uid: String, bestModel: Model[_], metrics: Array[Double]): TrainValidationSplitModel = {
    copyValues(new TrainValidationSplitModel(uid, bestModel, metrics).setParent(this))
  }
}

/**
 * :: Experimental ::
 * Model from train validation split.
 */
@Experimental
class TrainValidationSplitModel private[ml] (
    uid: String,
    bestModel: Model[_],
    avgMetrics: Array[Double])
  extends ValidationModel[TrainValidationSplitModel](uid, bestModel, avgMetrics) with TrainValidationSplitParams {

  override def copy(extra: ParamMap): TrainValidationSplitModel = {
    val copied = new TrainValidationSplitModel (
      uid,
      bestModel.copy(extra).asInstanceOf[Model[_]],
      avgMetrics.clone())
    copyValues(copied, extra)
  }
}