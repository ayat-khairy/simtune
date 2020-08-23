import math
import time
import os
from schema_30  import *
import subprocess, sys

current_milli_time = lambda: int(round(time.time() * 1000))
conf_filePath = "/local/data/HiBench/conf/spark-defaults.conf"
conf_cost_filePath = "/local/data/experiment/tpch_20_multitasking_from_PR_5m.csv"
conf_header= ""
#header_writen = False

#conf_names = ["spark.executor.cores" , "spark.executor.instances" , "MB.spark.memory.offHeap.size" , "Size.spark.io.compression.snappy.blockSize" , "Time.spark.locality.wait" , "spark.memory.fraction"]

intConfRange = {

"spark.executor.cores": [3,16],
"spark.executor.memory": [5, 43],
"spark.executor.instances": [8 , 48 ],
"spark.default.parallelism": [8, 50],
"MB.spark.memory.offHeap.size":[10 , 1000],    # in MB
"Size.spark.shuffle.file.buffer":[2, 128],
"MB.spark.reducer.maxSizeInFlight": [2,128],       # in MB
"spark.shuffle.sort.bypassMergeThreshold": [100,1000],
"MS.spark.speculation.interval": [100,1000],  #in ms
"MB.spark.broadcast.blockSize" : [2, 128],   # in MB
"Size.spark.io.compression.lz4.blockSize" : [2,128],
"Size.spark.io.compression.snappy.blockSize": [2,128],
"MB.spark.kryoserializer.buffer.max": [512, 2048], # in MB
"Size.spark.kryoserializer.buffer": [2,128],
"MB.spark.storage.memoryMapThreshold": [50,500], #in MB
#"spark.akka.failure.detector.threshold" : [100, 500],
#"spark.akka.heartbeat.pauses": [1000, 10000],
#"spark.akka.heartbeat.interval" : [200,5000],
#"spark.akka.threads": [1,8],
"Time.spark.network.timeout": [1300000, 1400000] , #in seconds
##"Time.spark.files.fetchTimeout": [20, 500], #in seconds
"Time.spark.locality.wait": [1,10], #in seconds
##"Time.spark.scheduler.revive.interval" : [2,50],
##"spark.task.maxFailures" : [1,8],


}
floatConfRange ={
"spark.memory.fraction":[0.5 , 1],
"spark.memory.storageFraction": [0.5 , 1],
"spark.speculation.multiplier":[1,5],
"spark.speculation.quantile": [0,1],
}

def execute_run(params):
    print ("executing a new run .... ")

    param_names = params.keys()
    print (">>>>>> param names >>> " , param_names)


   ##### set the insig int and float params to mean
    for param in intConfRange.keys():
        if param not in param_names:
            params [param] = [(intConfRange[param][0] + intConfRange[param][1])/2] ### mean value

    for param in floatConfRange.keys():
        if param not in param_names:
            params [param] = [(floatConfRange[param][0] + floatConfRange[param][1])/2.0] ### mean value

    param_names = params.keys()
    print (">>>>>> param names >>> " , param_names)

    param_names.sort() # for logging purposes, to keep the same parameters order over the different executions
    print (">>>>>> param names >>> " , param_names)
    ##########################################

    print ("executing a new run .... ")
    conf_cost = ""
    global conf_header
    for key , value in params.iteritems():
       conf_header = conf_header + "," +  key
    conf_header = conf_header + "\n"
    #write_conf (conf_cost_filePath , conf_header, 'a')
    confs = ""
    for key, value in params.iteritems():
       print (">>>>>>>>>>>>>>>> param >>> " ,  key , "  >>> " , params[key])
       conf_cost = conf_cost + str(params[key][0])+ ","
       if "Size"  in key:
	       confs = confs + key[5:] + " " + str (params[key][0]) + "k \n"
       elif  "spark.executor.memory"  in key:
               confs = confs + key + " " + str(params[key][0]) + "g \n"

       else:
               confs = confs + str(key) + " " + str (params[key][0]) + "\n"

    write_conf(conf_filePath , confs , 'w') 
    conf = Configuration()
   # conf.spark_executor_memory = params ['spark.executor.memory'][0]
   # conf.spark_speculation = params['spark.speculation'][0]
   # conf.spark_reducer_maxSizeInFlight = params ['MB.spark.reducer.maxSizeInFlight'][0]
   # conf.spark_rdd_compress= params ['spark.rdd.compress'][0]
   # conf.spark_executor_cores = params ['spark.executor.cores'][0]
  #  conf.spark_memory_storageFraction= params ['spark.memory.storageFraction'][0]
  #  conf.spark_memory_fraction= params['spark.memory.fraction'][0]
#    conf.spark_shuffle_sort_bypassMergeThreshold = params ['spark.shuffle.sort.bypassMergeThreshold'][0]
#    conf.spark_executor_instances= params ['spark.executor.instances'][0]
    #conf.spark_broadcat_blockSize=39385
   # conf.spark_default_parallelism= params ['hibench.default.map.parallelism'][0]
 #   conf.spark_memory_offHeap_size = params ['MB.spark.memory.offHeap.size'][0]
  #  conf.spark_locality_wait = params ['Time.spark.locality.wait'][0]
  #  conf.spark_io_compression_snappy_blockSize = params ['Size.spark.io.compression.snappy.blockSize'][0]
  #  conf.spark_shuffle_file_buffer= params ['Size.spark.shuffle.file.buffer'][0]
    #conf.spark_shuffle_spill_compress = False
    #conf.spark_shuffle_spill = True
    #conf.spark_memory_offHeap_enabled= params ['spark.memory.offHeap.enabled'][0]
    #conf.spark_speculation = params ['spark.speculation'][0]
     #  conf.spark_app_name= "LogisticRegressionWithLBFGS"
   
    conf = create_conf(params)

    conf.spark_app_name= "tpch"
    conf.input_size = 20
    #conf.spark_app_name= "kmeans"

    #spark_cmd = 'sudo /usr/lib/spark/bin/spark-submit  --master yarn-client  --conf spark.app.name=\"tpc-h\"  --conf spark.driver.memory=35g '
    confs = ""
    for key, value in params.items():
       #spark_conf = spark_conf + str (cfg[param]) + ","
       if ("MB." in str(key)):
           confs += '{0}={1}m \n'.format( key[3:], value[0])
       elif ("MS." in str(key)):
           confs += '{0}={1}ms \n'.format( key[3:], value[0])
       elif ("Time." in str(key)):
           confs += '{0}={1}s \n'.format( key[5:], value[0])
       elif ("Size." in str(key)):
           confs += '{0}={1}k \n'.format( key [5:], value[0])
       elif (key == 'spark.executor.memory'):
           confs += '{0}={1}g \n'.format( key, value[0])
       else:
           confs += '{0}={1} \n'.format( key, value[0])
    print (">>>>> confs >>>> " , confs)
    write_conf (conf_filePath ,confs , 'w')   
    
    spark_cmd = "sudo /usr/lib/spark/bin/spark-submit  --master yarn-client --conf spark.eventLog.enabled=true  --conf spark.eventLog.dir=hdfs:///var/log/spark/apps  --conf spark.app.name=\"tpc-h\"  --conf spark.driver.memory=35g --properties-file " + conf_filePath +  " --jars /local/data/hadoop-gpl-compression-0.1.0/hadoop-gpl-compression-0.1.0.jar --class \"main.scala.TpchQuery\"  /local/data/code/tpch/tpch-spark/target/scala-2.11/spark-tpc-h-queries_2.11-1.0.jar"



    exec_time = getCost (conf)
    if (exec_time == 1000000000 ):
        print (">>>>>>>>>>>>>>>>> cost not found in DB ....")    
    	start = current_milli_time ()
         #  command = " sudo /local/data/HiBench/bin/workloads/ml/lr/spark/run.sh"
#        command = "sudo /local/data/HiBench/bin/workloads/websearch/pagerank/spark/run.sh"
	 #   command = "sudo /local/data/HiBench/bin/workloads/ml/kmeans/spark/run.sh"

 


        print (">>>>> " , spark_cmd)
   	p = subprocess.Popen(spark_cmd, shell=True, stderr=subprocess.PIPE)
 
	## But do not wait till netstat finish, start displaying output immediately ##
       	while True:
       		  out = p.stderr.read(1)
        	  if out == '' and p.poll() != None:
          	    break
       		  if out != '':
        	    sys.stdout.write(out)
         	    sys.stdout.flush()
	p.wait()
   	exec_time = current_milli_time() - start
    conf_cost = conf_cost+ str (exec_time) + "\n"
    write_conf (conf_cost_filePath ,conf_cost , 'a')
    insert_conf (conf , exec_time)
    return exec_time  # returning time as a cost


def create_conf (params):
    conf = Configuration()
    if 'spark.executor.memory' in params:
        conf.spark_executor_memory = params ['spark.executor.memory'][0]
    if 'spark.shuffle.compress' in params:
        conf.spark_shuffle_compress= params ['spark.shuffle.compress'][0]
    if 'spark.rdd.compress' in params:
        conf.spark_rdd_compress=params ['spark.rdd.compress'][0]
    if 'spark.executor.cores' in params:
        conf.spark_executor_cores = params ['spark.executor.cores'][0]
    if 'spark.memory.storageFraction' in params:
        conf.spark_memory_storageFraction= params['spark.memory.storageFraction'][0]
    if 'spark.memory.fraction' in params:
        conf.spark_memory_fraction= params['spark.memory.fraction'][0]
    if 'spark.executor.instances' in params:
        conf.spark_executor_instances= params ['spark.executor.instances'][0]
    if 'MB.spark.broadcast.blockSize' in params:
        conf.spark_broadcat_blockSize= params ['MB.spark.broadcast.blockSize'][0]
    if 'hibench.default.map.parallelism' in params:
        conf.spark_default_parallelism= params ['hibench.default.map.parallelism'][0]
    if 'MB.spark.memory.offHeap.size' in params:
        conf.spark_memory_offHeap_size = params ['MB.spark.memory.offHeap.size'][0]
    if 'Size.spark.shuffle.file.buffer' in params:
        conf.spark_shuffle_file_buffer=params ['Size.spark.shuffle.file.buffer'][0]
    if 'spark.shuffle.spill.compress' in params:
        conf.spark_shuffle_spill_compress = params ['spark.shuffle.spill.compress'][0]
    if 'spark.memory.offHeap.enabled' in params:
        conf.spark_memory_offHeap_enabled= params ['spark.memory.offHeap.enabled'][0]
    if 'spark.speculation' in params:
        conf.spark_speculation = params ['spark.speculation'][0]
    # ###########
    if 'MB.spark.reducer.maxSizeInFlight' in params:
        conf.spark_reducer_maxSizeInFlight=params ["MB.spark.reducer.maxSizeInFlight"][0]
    if 'spark.shuffle.sort.bypassMergeThreshold' in params:
        conf.spark_shuffle_sort_bypassMergeThreshold=params ["spark.shuffle.sort.bypassMergeThreshold" ][0]
    if 'MS.spark.speculation.interval' in params:
        conf.spark_speculation_interval=params ["MS.spark.speculation.interval"][0]
    if 'spark.speculation.multiplier' in params:
        conf.spark_speculation_multiplier=params ["spark.speculation.multiplier"][0]
    if 'spark.speculation.quantile' in params:
        conf.spark_speculation_quantile=params ["spark.speculation.quantile" ][0]
    if 'spark.io.compression.codec' in params:
        conf.spark_io_compression_codec= params ["spark.io.compression.codec" ][0]
    if 'Size.spark.io.compression.lz4.blockSize' in params:
        conf.spark_io_compression_lz4_blockSize=params ["Size.spark.io.compression.lz4.blockSize"][0]
    if 'Size.spark.io.compression.snappy.blockSize' in params:
        conf.spark_io_compression_snappy_blockSize=params ["Size.spark.io.compression.snappy.blockSize"][0]
    if 'spark.kryo.referenceTracking' in params:
        conf.spark_kryo_referenceTracking=params ["spark.kryo.referenceTracking"][0]
    if 'MB.spark.kryoserializer.buffer.max' in params:
        conf.spark_kryoserializer_buffer_max=params ["MB.spark.kryoserializer.buffer.max"][0]
    if 'MB.spark.storage.memoryMapThreshold' in params:
        conf.spark_storage_memoryMapThreshold = params ["MB.spark.storage.memoryMapThreshold"][0]
    if 'Size.spark.kryoserializer.buffer' in params:
        conf.spark_kryoserializer_buffer=params ["Size.spark.kryoserializer.buffer" ][0]
    if 'Time.spark.network.timeout' in params:
        conf.spark_network_timeout=params ["Time.spark.network.timeout"][0]
    if 'Time.spark.locality.wait' in params:
        conf.spark_locality_wait=params ["Time.spark.locality.wait"][0]
    if 'spark.broadcast.compress' in params:
        conf.spark_broadcast_compress=params ["spark.broadcast.compress"][0]
    if 'spark.serializer' in params:
        conf.spark_serializer=params ["spark.serializer"][0]

    return conf


def write_conf (conf_filePath , confs, mode):
  
  # global header_writen     
   writer = open (conf_filePath , mode)
  # if 'header_writen' not in write_conf._dict_ :
   #   writer.write (conf_header)
    #  write_conf.header_writen = True  
   writer.write (confs)
   writer.close()


def main(job_id, params):
    cost = execute_run (params)
#    print ("SparkTuner >> config >>> " , params[0])
    print ("SparkTuner>> cost >> " , cost)
    return cost

