package org.apache.spark.ml.tuning

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, ParamValidators, IntParam}
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

  private val f2jBLAS = new F2jBLAS

  /** @group setParam */
  def setTrainRatio(value: Double): this.type = set(trainRatio, value)

  override def fit(dataset: DataFrame): TrainValidationSplitModel = {
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sqlCtx = dataset.sqlContext
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val splits = MLUtils.sample(dataset.rdd, 2, 3d/4d, 4d/4d)
    val training = splits._1
    val validation = splits._2

    val trainingDataset = sqlCtx.createDataFrame(training, schema).cache()
    val validationDataset = sqlCtx.createDataFrame(validation, schema).cache()

    val metrics = measureModels(trainingDataset, validationDataset, est, eval, epm, numModels)

    f2jBLAS.dscal(numModels, 1.0, metrics, 1)
    logInfo(s"Average validation metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) = metrics.zipWithIndex.maxBy(_._1)
    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
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