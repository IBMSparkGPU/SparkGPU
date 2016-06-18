package org.apache.spark.ml.regression

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ DataFrame, Row }
import org.apache.spark.sql.functions.{ col, lit }
import org.apache.spark.sql.types.DoubleType

import org.apache.spark.ml.optim._

class LinearRegressionBatch(batch: Array[LinearRegression]) {

  def train(dataset: DataFrame, num_partition: Int = 2): RDD[(LinearRegressionModel, Double, Array[Double])] = {

    val start_time = System.currentTimeMillis()
    //assume all batches have the same label/feature col names
    val label_col = batch(0).getLabelCol
    val features_col = batch(0).getFeaturesCol
    
    val weighted = batch(0).isDefined(batch(0).weightCol)
    
    val weight_col = if(weighted) batch(0).getWeightCol else ""
    
    val intercept = batch(0).getFitIntercept
    
    val dataset_rdd = (
        if(weighted) dataset.select(col(label_col).cast(DoubleType), col(weight_col), col(features_col))
        else         dataset.select(col(label_col).cast(DoubleType), col(features_col))        
    ).rdd

    dataset_rdd.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    dataset_rdd.conf.registerKryoClasses(Array(classOf[LinearRegression], classOf[LinearRegressionModel], classOf[Row]))

    val batch_rdd = dataset_rdd.context.parallelize(batch, num_partition)
    
    val x = dataset_rdd.collect()

    val dataset_arr = dataset_rdd.context.broadcast(x)

    val lib_gpu = CuAccManager.get_lib_gpu(dataset_rdd.conf)
    val onset = dataset_rdd.conf.get("spark.ml.onset", "") == "y"
    val sparse = dataset_rdd.conf.get("spark.ml.sparse", "") == "y"

    require(!lib_gpu.isEmpty(), { printf("ERROR] empty GPU lib: LinearRegressionBatch is enabled on CuAcc-GPU only") })

    printf("batch initialization took %dms\n", System.currentTimeMillis() - start_time)

    val ret = batch_rdd.mapPartitionsWithIndex(
      (index, iterator) =>
        {
          var libs = lib_gpu.split(";")
          for (i <- 0 until libs.length)
            System.load(libs(i))

          val (handle, mbs, num_feature, eid, data_mean, data_std, lable_mean, lable_std, histogram, weight_sum) =
            CuAccManager.initialize_batch(index, dataset_arr.value, "LinearRegression", onset, sparse, weighted, intercept)

          val lor_batch = iterator.toArray

          val lor_model = lor_batch.zipWithIndex.map {
            case (b, iindex) =>
              {
                require(b.getLabelCol == label_col, { printf("inconsistent label column names %s vs %s\n", b.getLabelCol, label_col) })
                require(b.getFeaturesCol == features_col, { printf("inconsistent label column names %s vs %s\n", b.getFeaturesCol, features_col) })

                b.train_batch(eid + "." + iindex.toString(), handle, weight_sum, data_mean, data_std, lable_mean, lable_std)
              }
          }

          CuAccManager.destroy_batch(handle)

          Iterator(lor_model)
        }).flatMap(_.to)

    ret
  }

  def summarize(dataset: DataFrame, model: Array[(LinearRegressionModel, Double, Array[Double])]) = {

    val label_col = model(0)._1.getLabelCol
    val features_col = model(0)._1.getFeaturesCol

    model.foreach(m => {
      val (summaryModel, predictionColName) = m._1.findSummaryModelAndPredictionCol()

      val transformed_dataset = summaryModel.transform(dataset)

      //      transformed_dataset.rdd.saveAsTextFile("yyy.txt")

      val linRegSummary = new LinearRegressionTrainingSummary(
        transformed_dataset,
        predictionColName,
        label_col,
		features_col,
        m._1,
        Array(0D),        
        m._3)

      printf("rmse = %.16f (%.16f)\n", math.sqrt(linRegSummary.meanSquaredError), m._2)
    })
  }
}
