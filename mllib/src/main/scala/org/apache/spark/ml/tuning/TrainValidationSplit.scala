package org.apache.spark.ml.tuning

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{ParamMap, ParamValidators, IntParam}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame

/**
 * Params for [[TrainValidationSplit]] and [[TrainValidationSplitModel]].
 */
private[ml] trait TrainValidationSplitParams extends ValidationParams

@Experimental
class TrainValidationSplit(override val uid: String) extends Validation[TrainValidationSplitModel]
  with TrainValidationSplitParams with Logging {

  def this() = this(Identifiable.randomUID("cv"))

  private val f2jBLAS = new F2jBLAS

  override def fit(dataset: DataFrame): TrainValidationSplitModel = {
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
  }

  override def copy(extra: ParamMap): TrainValidationSplit = {
    val copied = defaultCopy(extra).asInstanceOf[TrainValidationSplit]
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
 * Model from k-fold cross validation.
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