/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.optim

class NativeCuAcc {
  @native def create_updater(
      name                : String, 
      option              : Int
      ): (Long)  //return handle    
   
  @native def destroy_updater(
      handle              : Long
      )     
      
  @native def updater_initialize(        
      handle              : Long, 
      weights             : Array[Double],  
      regParam            : Double,
      option              : Int
      ): (Double) //return regularization value

  @native def updater_convergence_compute(        
      handle              : Long, 
      weights             : Array[Double], //input weight to be updated in-place
      gradientSum         : Array[Double],
      miniBatchSize       : Long,
      stepSize            : Double,
      iter                : Int,
      regParam            : Double,
      convergenceInfo     : Array[Double],//return convergence info
      option              : Int
      ): (Double) //regularization value
  
  @native def get_exec_id(
      ): (String) 
            
      
  @native def create_acc_cluster(
      name                 : String,
      data                 : Array[Double],
      label                : Array[Double],
      option               : Int
      ) : (Long)   
  
  @native def create_sparse_acc_cluster(
      name                 : String,
      data                 : Array[Double],
      label                : Array[Double],
      ridx                 : Array[Int],
      cidx                 : Array[Int],
      num_feature          : Int,
      intercept            : Boolean,
      option               : Int
      ) : (Long)  

  @native def reset_acc_cluster(
      handle               : Long
      )
      
  @native def destroy_acc_cluster(
      handle               : Long
      ) : (Int)  
  
  @native def acc_cluster_summarize(
      handle               : Long,
      weight_sum           : Double,
      data_sum             : Array[Double],
      data_sq_sum          : Array[Double],
      label_info           : Array[Double]
      ) : (Int)
      
  @native def acc_cluster_weighten(
      handle               : Long, 
      weight               : Array[Double],
      weight_sum           : Array[Double]
      ) : (Int)             
  
  @native def acc_cluster_standardize(
      handle               : Long, 
      data_mean            : Array[Double],
      data_std             : Array[Double],
      label_mean           : Double,
      label_std            : Double
      ) : (Int)     
  
  @native def acc_cluster_evaluate(
      handle               : Long,
      miniBatchFraction    : Double,//sampling
      weight_sum           : Double, 
      weights              : Array[Double],
      cumGradient          : Array[Double],
      loss                 : Array[Double],//return loss    
      option               : Int
      ) : (Int)  
      
  @native def acc_cluster_aug(
      handle               : Long,
      weights              : Array[Double],
      intercept            : Double,
      metric               : Array[Double]//return metric    
      )
      
  @native def acc_cluster_rmse(
      handle               : Long,
      weights              : Array[Double],
      intercept            : Double,
      metric               : Array[Double]//return metric    
      )

  @native def compress(
      weights              : Array[Double],
      input                : Array[Int],
      threshold            : Double,
      option               : Int
      ) : Array[Int]
  
  @native def decompress(      
      weights              : Array[Double],
      input                : Array[Int],
      threshold            : Double,
      option               : Int
      )
}

object NativeCuAcc {
  def apply() = new NativeCuAcc
}