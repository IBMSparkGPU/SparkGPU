package org.apache.spark.ml.optim

import scala.collection.mutable
import java.io._

import breeze.linalg.{ DenseVector => BDV }
import breeze.optimize.{ CachedDiffFunction, DiffFunction }
import _root_.org.apache.spark.ml.linalg._
import _root_.org.apache.spark.rdd.RDD
import _root_.org.apache.spark.sql.{ DataFrame, Row }
import _root_.org.apache.spark.storage.StorageLevel
import org.apache.spark.{ SparkConf, SparkContext }


//import org.apache.spark.mllib.linalg.{ Vector, Vectors, VectorUDT }

// Import Spark SQL data types
import org.apache.spark.sql.types.{ StructType, StructField, StringType, DoubleType };


@SerialVersionUID(1209L)
class ExecElem(handle: Long, weight_sum: Double, weight_nnz: Double, num_feature: Int, eid: String) extends Serializable {

  val m_handle = handle
  val m_weight_sum = weight_sum
  val m_weight_nnz = weight_nnz
  val m_num_feature = num_feature
  val m_eid = eid

  def print(index: Int) = printf("%s.%d] h=%d weight_sum=%f weight_nnz=%f numf=%d\n", m_eid, index, m_handle, m_weight_sum, m_weight_nnz, m_num_feature)
}

object CuAccManager {

  var num_executor = 0
  var lib_loaded = false

  def get_lib_gpu(conf: SparkConf): String = conf.get("spark.ml.useGPU", "")

  def load_library(exec_rdd: RDD[Int], libs: Array[String]) = {
    if (!lib_loaded) {
      val eid_list = exec_rdd.mapPartitionsWithIndex {
        (index, iterator) =>
          {
            for (i <- 0 until libs.length)
              System.load(libs(i))
          }
          Iterator(NativeCuAcc().get_exec_id())
      }.distinct().collect()

      eid_list.foreach { printf("--------lib loaded in %s--------\n", _) }
      lib_loaded = true
    }
  }

  def get_num_executor(context: SparkContext, lib_gpu: String): Int =
    {
      if (num_executor == 0) {
        val exec_rdd = context.parallelize(0 to 64, 64)
        load_library(exec_rdd, lib_gpu.split(";"))

        num_executor = exec_rdd.mapPartitionsWithIndex {
          (index, iterator) =>
            {}
            Iterator(NativeCuAcc().get_exec_id())
        }.distinct().count().toInt

        printf("num_executor = %d\n", num_executor)
      }
      num_executor
    }

  def read(filename: String, sc: SparkContext): DataFrame =
    {
      val _file = sc.textFile(filename)
      val sqlContext = new org.apache.spark.sql.SQLContext(sc)

      val _rdd = _file.map(_.split(",")).map(p =>
        {
          val label = p(0).trim.drop(1).toDouble
          val weight = p(1).trim.toDouble
          var num_feature = p(2).trim.drop(1).toInt

          if (p(3) == "[]") {
            var indices = Array.fill(0) { 0 }
            var values = Array.fill(0) { 1.0 }

            val sv = Vectors.sparse(num_feature, indices, values)
            Row(label, weight, sv)
          } else {

            var begin = 3

            def get_indice: Int = {
              var end = 0
              for (end <- begin to num_feature) {
                val last = p(end).takeRight(1)
                if (last == "]") {
                  return end
                }
              }

              return -1
            }

            val end = get_indice

            val num_nnz = end - begin + 1

            var indices = Array.fill(num_nnz) { 0 }
            var values = Array.fill(num_nnz) { 1.0 }
            var i = 0
            for (i <- begin until end + 1) {
              val index = p(i)
              var clean = index.replaceAll("\\[", "").replaceAll("\\]", "")

              if (clean.isEmpty()) {
                clean = "0"
              } else {
                indices(i - begin) = clean.toInt
              }
            }

            val sv = Vectors.sparse(num_feature, indices, values)

            Row(label, weight, sv)
          }

        })

      var schema_field = Array(StructField("label", DoubleType, true), StructField("weight", DoubleType, true), StructField("features", new VectorUDT, true))

      val df = sqlContext.createDataFrame(_rdd, StructType(schema_field))

      df.printSchema

      df
    }

  def dump(dataset_rdd: RDD[Row], filename: String) =
    {
      printf("dumping %s\n", filename)

      val writer = new PrintWriter(new File(filename))
      dataset_rdd.collect().foreach { x =>
        {
          val feature_idx = x.size
          val l = x.getDouble(0)
          val sv = x.getAs[Vector](feature_idx - 1).toSparse

          writer.write(l.toString())

          sv.foreachActive((i, v) => {
            writer.write(" " + (i + 1) + ":" + v)
          })

          if (feature_idx > 2)
            writer.write(" w=" + x.getDouble(1).toString())

          writer.write("\n")
        }
      }

      writer.close()
    }

  def initialize_batch(index: Int, dataset_arr: Array[Row], algo: String, onset: Boolean, sparse: Boolean, weighted: Boolean, intercept: Boolean): (Long, Double, Int, String, Array[Double], Array[Double], Double, Double, Array[Double], Double) =
    {
      var exec_elem = CuAccManager.create_batch(dataset_arr,
        algo, onset, sparse, weighted,
        intercept)

      exec_elem.print(index)

      val (data_mean, data_sq_sum, label_info) = CuAccManager.summarize_batch(exec_elem.m_handle, exec_elem.m_num_feature, exec_elem.m_weight_sum)

      val unbiased_factor = exec_elem.m_weight_nnz / (exec_elem.m_weight_nnz - 1)
      val _data_sq_sum = Vectors.dense(data_sq_sum.toArray)

      //mean and unbiased sample variance/_data_std    
      _data_sq_sum.foreachActive {
        case (index, value) => {
          data_sq_sum(index) = math.sqrt((value - data_mean(index) * data_mean(index)) * unbiased_factor)
        }
      }
      val label_mean = label_info(1) / exec_elem.m_weight_sum
      val label_std = math.sqrt((label_info(2) / exec_elem.m_weight_sum - label_mean * label_mean) * unbiased_factor)

      CuAccManager.standardize_batch(exec_elem.m_handle, data_mean.toArray, data_sq_sum.toArray, label_mean, if (algo == "LogisticRegression") 1 else label_std)

      (exec_elem.m_handle, exec_elem.m_weight_sum, exec_elem.m_num_feature, exec_elem.m_eid, data_mean.toArray, data_sq_sum.toArray, label_mean, label_std, label_info.toArray, exec_elem.m_weight_sum)
    }
	 
  def create_batch(dataset_arr: Array[Row], algo: String, onset: Boolean, sparse: Boolean, weighted: Boolean, intercept: Boolean): ExecElem = {

    //assume: only two columns in the row: label, vector. NO WEIGHT SUPPORT YET
    var handle = 0L
    var mbs = 0L
    val feature_idx = dataset_arr(0).size - 1 //the last column is the feature, the first column is the label, ignore any in-between except for weight
    val weight_idx = feature_idx - 1
    val features = dataset_arr(0).getAs[Vector](feature_idx)
    val num_feature = features.size;

    require(weight_idx > 0 || (!weighted), { printf("cannot located weight index (weight_idx=%d)\n", weight_idx) })

    var _label = new Array[Double](dataset_arr.length)
    var _weight = if (weighted) new Array[Double](dataset_arr.length) else null
    
    if (sparse && features.getClass.getSimpleName == "SparseVector") {

      var tot_nnz = 0;

      dataset_arr.foreach(v => {
        val sv = v.getAs[Vector](feature_idx).toSparse
        tot_nnz += sv.numActives
      })

      var _ridx = new Array[Int](dataset_arr.length + 1)
      var _cidx = new Array[Int](tot_nnz)
      var _data = if (onset) null else new Array[Double](tot_nnz)
      var idx = 0
      dataset_arr.foreach(v => {
        //stack up v for gpu              
        val vec = v.getAs[Vector](feature_idx).toSparse

        Array.copy(vec.indices, 0, _cidx, idx, vec.numActives)

        if (!onset) Array.copy(vec.values, 0, _data, idx, vec.numActives)

        if (weighted) _weight(mbs.toInt) = v.getDouble(weight_idx)

        _ridx(mbs.toInt) = idx
        _label(mbs.toInt) = v.getDouble(0)

        idx += vec.numActives
        mbs += 1
      })

      _ridx(mbs.toInt) = idx

      handle = NativeCuAcc().create_sparse_acc_cluster(algo, _data, _label, _ridx, _cidx, num_feature, intercept, 0)

    } else {

      var _data = new Array[Double](dataset_arr.length * num_feature)

      dataset_arr.foreach(v => {
        //stack up v for gpu        
        val vec = v.getAs[Vector](feature_idx)

        Array.copy(vec.toArray, 0, _data, mbs.toInt * num_feature, num_feature)

        if (weighted) _weight(mbs.toInt) = v.getDouble(weight_idx)

        _label(mbs.toInt) = v.getDouble(0)

        mbs += 1
      })

      handle = NativeCuAcc().create_acc_cluster(algo, _data, _label, 0)
    }

    var _weight_info = Array.fill(2) { 0.0 }
    NativeCuAcc().acc_cluster_weighten(handle, _weight, _weight_info)
    NativeCuAcc().reset_acc_cluster(handle)

    new ExecElem(handle, _weight_info(0), _weight_info(1), num_feature, NativeCuAcc().get_exec_id())
  }

  def initialize(dataset_rdd: RDD[Row], lib_gpu: String, algo: String, weighted: Boolean, intercept: Boolean): (Array[ExecElem], RDD[Int], Array[Double], Array[Double], Double, Double, Array[Double], Double) = {

    //create execution info
    val (exec_info, exec_rdd) = create(dataset_rdd,
      lib_gpu.split(";"), algo,
      dataset_rdd.conf.get("spark.ml.onset", "") == "y",
      dataset_rdd.conf.get("spark.ml.sparse", "") == "y",
      weighted,
      intercept)

    exec_info.zipWithIndex.foreach {
      case (v, index) => v.print(index)
    }

    var weight_sum = 0.0
    var weight_nnz = 0.0
    exec_info.foreach(v => {
      weight_sum += v.m_weight_sum
      weight_nnz += v.m_weight_nnz
    })

    var (data_mean, data_sq_sum, label_info, num_exec) = summarize(exec_rdd, exec_info, weight_sum)
    reset(exec_rdd, exec_info, num_exec)

    val unbiased_factor = weight_nnz / (weight_nnz - 1)
    val _data_sq_sum = Vectors.dense(data_sq_sum.toArray)

    //mean and unbiased sample variance/_data_std    
    _data_sq_sum.foreachActive {
      case (index, value) => {
        data_sq_sum(index) = math.sqrt((value - data_mean(index) * data_mean(index)) * unbiased_factor)
      }
    }

    val label_mean = label_info(1) / weight_sum
    val label_std = math.sqrt((label_info(2) / weight_sum - label_mean * label_mean) * unbiased_factor)
    num_exec = standardize(exec_rdd, exec_info, data_mean.toArray, data_sq_sum.toArray, label_mean, label_std)
    reset(exec_rdd, exec_info, num_exec)

    (exec_info, exec_rdd, data_mean.toArray, data_sq_sum.toArray, label_mean, label_std, label_info.toArray, weight_sum)
  }

  def create(dataset_rdd: RDD[Row], libs: Array[String], algo: String, onset: Boolean, sparse: Boolean, weighted: Boolean, intercept: Boolean): (Array[ExecElem], RDD[Int]) = {

    //create execution rdd
    val num_partition = dataset_rdd.getNumPartitions
    val exec_rdd = dataset_rdd.context.parallelize(0 to num_partition, num_partition)

    load_library(exec_rdd, libs)

    val exec_info = dataset_rdd.mapPartitionsWithIndex {
      (index, iterator) =>
        {
          val start_time = System.currentTimeMillis()
          var exec_elem = create_batch(iterator.toArray, algo, onset, sparse, weighted, intercept)
          printf("*****create took %dms\n", System.currentTimeMillis() - start_time)
          Iterator(exec_elem)
        }
    }.collect()

    (exec_info, exec_rdd)
  }

  def summarize_batch(handle: Long, num_feature: Int, weight_sum: Double): (BDV[Double], BDV[Double], BDV[Double]) = {

    var _data_sum = Array.fill(num_feature) { 0.0 }
    var _data_sq_sum = Array.fill(num_feature) { 0.0 }
    var _label_info = Array.fill(3) { 0.0 }

    require(NativeCuAcc().acc_cluster_summarize(handle, weight_sum, _data_sum, _data_sq_sum, _label_info) == 1, { printf("$$$ERROR  num_exec") })
    NativeCuAcc().reset_acc_cluster(handle)

    (BDV(_data_sum), BDV(_data_sq_sum), BDV(_label_info))
  }

  def summarize(exec_rdd: RDD[Int], exec_info: Array[ExecElem], weight_sum: Double): (BDV[Double], BDV[Double], BDV[Double], Int) = {

    exec_rdd.mapPartitionsWithIndex {
      (index, iterator) =>
        {
          val _eid = NativeCuAcc().get_exec_id()
          val _num_feature = exec_info(0).m_num_feature

          var _data_sum = Array.fill(_num_feature) { 0.0 }
          var _data_sq_sum = Array.fill(_num_feature) { 0.0 }
          var _label_info = Array.fill(3) { 0.0 } //(0): number_label-sum (1): sum, or number_of_one for binary label (2): sq_sum
          var _num_exec = 0

          exec_info.foreach(v => {
            if (v.m_eid == _eid) {
              _num_exec = _num_exec + NativeCuAcc().acc_cluster_summarize(v.m_handle, weight_sum, _data_sum, _data_sq_sum, _label_info)
            }
          })

          Iterator((BDV(_data_sum), BDV(_data_sq_sum), BDV(_label_info), _num_exec))
        }
    }.reduce((c1, c2) => (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3, c1._4 + c2._4))
  }

  def standardize_batch(handle: Long, data_mean: Array[Double], data_std: Array[Double], label_mean: Double, label_std: Double) = {

    require(NativeCuAcc().acc_cluster_standardize(handle, data_mean, data_std, label_mean, label_std) == 1, { printf("$$$ERROR  num_exec") })
    NativeCuAcc().reset_acc_cluster(handle)
  }

  def standardize(exec_rdd: RDD[Int], exec_info: Array[ExecElem], data_mean: Array[Double], data_std: Array[Double], label_mean: Double, label_std: Double): Int = {

    exec_rdd.mapPartitionsWithIndex {
      (index, iterator) =>
        {
          val _eid = NativeCuAcc().get_exec_id()
          var _num_exec = 0

          exec_info.foreach(v => {
            if (v.m_eid == _eid) {
              _num_exec += NativeCuAcc().acc_cluster_standardize(v.m_handle, data_mean, data_std, label_mean, label_std)
            }
          })

          Iterator(_num_exec)
        }
    }.reduce((c1, c2) => (c1 + c2))
  }

  def compute_aug_batch(handle: Long, weights: Array[Double], intercept: Double): Double = {

    var _metric = Array.fill(1) { 0.0 }

    NativeCuAcc().acc_cluster_aug(handle, weights, intercept, _metric)

    _metric(0)
  }

  def compute_rmse_batch(handle: Long, weights: Array[Double], intercept: Double): Double = {

    var _metric = Array.fill(1) { 0.0 }

    NativeCuAcc().acc_cluster_rmse(handle, weights, intercept, _metric)

    _metric(0)
  }

  def evaluate_batch(handle: Long, weight_sum: Double, weights: Array[Double]): (BDV[Double], Double) = {

    var _cgrad = Array.fill(weights.length) { 0.0 }
    var _loss = Array.fill(1) { 0.0 }

    require(NativeCuAcc().acc_cluster_evaluate(handle, 1.0, weight_sum, weights, _cgrad, _loss, 0) == 1, { printf("$$$ERROR  num_exec") })
    NativeCuAcc().reset_acc_cluster(handle)

    (BDV(_cgrad), _loss(0))
  }

  def evaluate(exec_rdd: RDD[Int], exec_info: Array[ExecElem], weight_sum: Double, weights: Array[Double]): (BDV[Double], Double, Int) = {

    exec_rdd.mapPartitionsWithIndex {
      (index, iterator) =>
        {
          val _eid = NativeCuAcc().get_exec_id()
          var _cgrad = Array.fill(weights.length) { 0.0 }
          var _loss = Array.fill(1) { 0.0 }
          var _num_exec = 0

          exec_info.foreach(v => {
            if (v.m_eid == _eid) {
              _num_exec += NativeCuAcc().acc_cluster_evaluate(v.m_handle, 1.0, weight_sum, weights, _cgrad, _loss, index)
            }
          })

          Iterator((BDV(_cgrad), _loss(0), _num_exec))
        }
    }.reduce((c1, c2) => (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3))
  }

  def reset(exec_rdd: RDD[Int], exec_info: Array[ExecElem], num_exec: Int) {

    require(exec_info.length == num_exec, { printf("$$$ERROR  num_exec=%d exec_info.length=%d\n", num_exec, exec_info.length) })

    var num_acc_cluster_processed = exec_rdd.mapPartitionsWithIndex {
      (index, iterator) =>
        {
          val _eid = NativeCuAcc().get_exec_id()

          exec_info.foreach(v => {
            if (v.m_eid == _eid) {
              NativeCuAcc().reset_acc_cluster(v.m_handle)
            }
          })

          Iterator(0)
        }
    }.count()
  }

  def destroy_batch(handle: Long) {

    require(NativeCuAcc().destroy_acc_cluster(handle) == 1, { printf("$$$ERROR  num_exec") })
  }

  def destroy(exec_rdd: RDD[Int], exec_info: Array[ExecElem]) {

    var num_exec = exec_rdd.mapPartitionsWithIndex {
      (index, iterator) =>
        {
          val _eid = NativeCuAcc().get_exec_id()
          var _num_exec = 0L

          exec_info.foreach(v => {
            if (v.m_eid == _eid) {
              _num_exec = _num_exec + NativeCuAcc().destroy_acc_cluster(v.m_handle)
            }
          })

          Iterator(_num_exec)
        }
    }.reduce((c1, c2) => (c1 + c2))

    require(exec_info.length == num_exec, { printf("$$$ERROR in destroy_acc_cluster num_exec=%d exec_info.length=%d\n", num_exec, exec_info.length) })
  }
}

class LogisticCostFun_CuAcc_Batch(
    eid: String,
    handle: Long,
    weight_sum: Double,
    fitIntercept: Boolean,
    standardization: Boolean,
    featuresStd: Array[Double],
    regParamL2: Double) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {

    val numFeatures = featuresStd.length
    val coeffs = Vectors.fromBreeze(coefficients)
    val start_time = System.currentTimeMillis();

    val (totalGradientArray, lossSum) = CuAccManager.evaluate_batch(handle, weight_sum, coeffs.toArray)

    // regVal is the sum of coefficients squares excluding intercept for L2 regularization.
    val regVal = if (regParamL2 == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.foreachActive { (index, value) =>
        // If `fitIntercept` is true, the last term which is intercept doesn't
        // contribute to the regularization.
        if (index != numFeatures) {
          // The following code will compute the loss of the regularization; also
          // the gradient of the regularization, and add back to totalGradientArray.
          sum += {
            if (standardization) {
              totalGradientArray(index) += regParamL2 * value
              value * value
            } else {
              if (featuresStd(index) != 0.0) {
                // If `standardization` is false, we still standardize the data
                // to improve the rate of convergence; as a result, we have to
                // perform this reverse standardization by penalizing each component
                // differently to get effectively the same objective function when
                // the training dataset is not standardized.
                val temp = value / (featuresStd(index) * featuresStd(index))
                totalGradientArray(index) += regParamL2 * temp
                value * temp
              } else {
                0.0
              }
            }
          }
        }
      }
      0.5 * regParamL2 * sum
    }

    if (totalGradientArray.length > numFeatures)
      printf("%s] GPU loss=%e totalGradientArray(0)=%e regVal=%e(L2=%e) intercept=%e\n", eid,
        lossSum / weight_sum, totalGradientArray(0), regVal, regParamL2, totalGradientArray(numFeatures))
    else
      printf("%s] GPU loss=%e totalGradientArray(0)=%e regVal=%e(L2=%e)\n", eid,
        lossSum / weight_sum, totalGradientArray(0), regVal, regParamL2)

    (lossSum / weight_sum + regVal, totalGradientArray)
  }
}

object LogisticCostFun_CuAcc {
  var num_called = 0
  var tot_duration = 0L
}

class LogisticCostFun_CuAcc(
    exec_rdd: RDD[Int],
    exec_info: Array[ExecElem],
    weight_sum: Double,
    fitIntercept: Boolean,
    standardization: Boolean,
    featuresStd: Array[Double],
    regParamL2: Double) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {

    val numFeatures = featuresStd.length
    val coeffs = Vectors.fromBreeze(coefficients)
    val start_time = System.currentTimeMillis();

    val (totalGradientArray, lossSum, num_exec) = CuAccManager.evaluate(exec_rdd, exec_info, weight_sum, coeffs.toArray)
    CuAccManager.reset(exec_rdd, exec_info, num_exec)

    LogisticCostFun_CuAcc.num_called += 1
    LogisticCostFun_CuAcc.tot_duration += System.currentTimeMillis() - start_time

    // regVal is the sum of coefficients squares excluding intercept for L2 regularization.
    val regVal = if (regParamL2 == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.foreachActive { (index, value) =>
        // If `fitIntercept` is true, the last term which is intercept doesn't
        // contribute to the regularization.
        if (index != numFeatures) {
          // The following code will compute the loss of the regularization; also
          // the gradient of the regularization, and add back to totalGradientArray.
          sum += {
            if (standardization) {
              totalGradientArray(index) += regParamL2 * value
              value * value
            } else {
              if (featuresStd(index) != 0.0) {
                // If `standardization` is false, we still standardize the data
                // to improve the rate of convergence; as a result, we have to
                // perform this reverse standardization by penalizing each component
                // differently to get effectively the same objective function when
                // the training dataset is not standardized.
                val temp = value / (featuresStd(index) * featuresStd(index))
                totalGradientArray(index) += regParamL2 * temp
                value * temp
              } else {
                0.0
              }
            }
          }
        }
      }
      0.5 * regParamL2 * sum
    }

    if (fitIntercept)
      printf("%d %d ms] GPU loss=%e totalGradientArray(0)=%e regVal=%e(L2=%e) intercept=%e\n",
        LogisticCostFun_CuAcc.num_called, LogisticCostFun_CuAcc.tot_duration,
        lossSum / weight_sum, totalGradientArray(0), regVal, regParamL2, totalGradientArray(numFeatures))
    else
      printf("%d %d ms] GPU loss=%e totalGradientArray(0)=%e regVal=%e(L2=%e)\n",
        LogisticCostFun_CuAcc.num_called, LogisticCostFun_CuAcc.tot_duration,
        lossSum / weight_sum, totalGradientArray(0), regVal, regParamL2)

    (lossSum / weight_sum + regVal, totalGradientArray)
  }
}

object LeastSquaresCostFun_CuAcc {
  var num_called = 0
  var tot_duration = 0L
}

/**
 * LeastSquaresCostFun_CuAcc implements Breeze's DiffFunction[T] for Least Squares cost.
 * It returns the loss and gradient with L2 regularization at a particular point (coefficients).
 * It's used in Breeze's convex optimization routines.
 */
class LeastSquaresCostFun_CuAcc(
    exec_rdd: RDD[Int],
    exec_info: Array[ExecElem],
    weight_sum: Double,
    base_offset: Double,
    fitIntercept: Boolean,
    standardization: Boolean,
    featuresStd: Array[Double],
    effectiveL2regParam: Double) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {

    val numFeatures = featuresStd.length
    val coeffs = Vectors.fromBreeze(coefficients)
    val start_time = System.currentTimeMillis();

    val (totalGradientArray, lossSum, num_exec) = CuAccManager.evaluate(exec_rdd, exec_info, weight_sum, coeffs.toArray)
    CuAccManager.reset(exec_rdd, exec_info, num_exec)

    LeastSquaresCostFun_CuAcc.num_called += 1
    LeastSquaresCostFun_CuAcc.tot_duration += System.currentTimeMillis() - start_time

    val regVal = if (effectiveL2regParam == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.foreachActive { (index, value) =>
        // The following code will compute the loss of the regularization; also
        // the gradient of the regularization, and add back to totalGradientArray.
        sum += {
          if (standardization) {
            totalGradientArray(index) += effectiveL2regParam * value
            value * value
          } else {
            if (featuresStd(index) != 0.0) {
              // If `standardization` is false, we still standardize the data
              // to improve the rate of convergence; as a result, we have to
              // perform this reverse standardization by penalizing each component
              // differently to get effectively the same objective function when
              // the training dataset is not standardized.
              val temp = value / (featuresStd(index) * featuresStd(index))
              totalGradientArray(index) += effectiveL2regParam * temp
              value * temp
            } else {
              0.0
            }
          }
        }
      }
      0.5 * effectiveL2regParam * sum
    }

    printf("%d %d ms] GPU loss=%e totalGradientArray(0)=%e regVal=%e(L2=%e)\n",
      LeastSquaresCostFun_CuAcc.num_called, LeastSquaresCostFun_CuAcc.tot_duration,
      lossSum / weight_sum, totalGradientArray(0), regVal, effectiveL2regParam)

    (lossSum / weight_sum + regVal, totalGradientArray)
  }
}

/**
 * LeastSquaresCostFun_CuAcc_Batch implements Breeze's DiffFunction[T] for Least Squares cost.
 * It returns the loss and gradient with L2 regularization at a particular point (coefficients).
 * It's used in Breeze's convex optimization routines.
 */
class LeastSquaresCostFun_CuAcc_Batch(
    eid: String,
    handle: Long,
    weight_sum: Double,
    base_offset: Double,
    fitIntercept: Boolean,
    standardization: Boolean,
    featuresStd: Array[Double],
    effectiveL2regParam: Double) extends DiffFunction[BDV[Double]] {

  override def calculate(coefficients: BDV[Double]): (Double, BDV[Double]) = {

    val numFeatures = featuresStd.length
    val coeffs = Vectors.fromBreeze(coefficients)
    val start_time = System.currentTimeMillis();

    val (totalGradientArray, lossSum) = CuAccManager.evaluate_batch(handle, weight_sum, coeffs.toArray)

    val regVal = if (effectiveL2regParam == 0.0) {
      0.0
    } else {
      var sum = 0.0
      coeffs.foreachActive { (index, value) =>
        // The following code will compute the loss of the regularization; also
        // the gradient of the regularization, and add back to totalGradientArray.
        sum += {
          if (standardization) {
            totalGradientArray(index) += effectiveL2regParam * value
            value * value
          } else {
            if (featuresStd(index) != 0.0) {
              // If `standardization` is false, we still standardize the data
              // to improve the rate of convergence; as a result, we have to
              // perform this reverse standardization by penalizing each component
              // differently to get effectively the same objective function when
              // the training dataset is not standardized.
              val temp = value / (featuresStd(index) * featuresStd(index))
              totalGradientArray(index) += effectiveL2regParam * temp
              value * temp
            } else {
              0.0
            }
          }
        }
      }
      0.5 * effectiveL2regParam * sum
    }

    printf("%s] GPU loss=%e totalGradientArray(0)=%e regVal=%e(L2=%e)\n", eid,
      lossSum / weight_sum, totalGradientArray(0), regVal, effectiveL2regParam)

    (lossSum / weight_sum + regVal, totalGradientArray)
  }
}