import os

#from sqlobject import *


import csv, sqlite3

db_path = "/local/data/experiment/tuneful.db"

def create_table():
    #csv_path = '/local/data/dataset/allappIds_15params_cost_cleaned.csv'
    #csv_path = '/local/data/experiment/allappIds_15params_cost_cleaned.csv'
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("CREATE TABLE workloads_30conf ( spark_executor_memory INTEGER ,spark_shuffle_compress TEXT ,spark_rdd_compress TEXT ,spark_executor_cores INTEGER ,spark_memory_storageFraction DECIMAL ,spark_memory_fraction DECIMAL ,spark_executor_instances INTEGER ,spark_broadcat_blockSize INTEGER ,spark_default_parallelism INTEGER ,spark_memory_offHeap_size INTEGER ,spark_shuffle_file_buffer INTEGER ,spark_shuffle_spill_compress TEXT ,spark_memory_offHeap_enabled TEXT ,spark_speculation TEXT , spark_reducer_maxSizeInFlight INTEGER, spark_shuffle_sort_bypassMergeThreshold INTEGER , spark_speculation_interval INTEGER , spark_io_compression_lz4_blockSize INTEGER, spark_io_compression_snappy_blockSize INTEGER, spark_kryoserializer_buffer_max INTEGER , spark_kryoserializer_buffer INTEGER , spark_storage_memoryMapThreshold INTEGER, spark_network_timeout INTEGER, spark_locality_wait INTEGER,     spark_speculation_multiplier DECIMAL , spark_speculation_quantile DECIMAL , spark_kryo_referenceTracking TEXT ,  spark_broadcast_compress TEXT ,  spark_io_compression_codec TEXT , spark_serializer TEXT  ,execution_time INTEGER, spark_app_name TEXT, spark_app_id TEXT , input_size INTEGER );") # use your column names here
    #reader = csv.reader(open(csv_path,'r'), delimiter=',')
    #next(reader, None) #skip header
   # for row in reader:
    #  to_db = [unicode(row[0], "utf8"), unicode(row[1], "utf8"), unicode(row[2], "utf8"),unicode(row[3], "utf8"), unicode(row[4],"utf8") , unicode(row[5]) , unicode(row[6]) ,unicode(row[7], "utf8") ,unicode(row[8], "utf8") ,unicode(row[9], "utf8") ,unicode(row[10], "utf8") ,unicode(row[11], "utf8") ,unicode(row[12], "utf8") ,unicode(row[13], "utf8") ,unicode(row[14], "utf8") ,
      
      
      
      
     # unicode(row[15], "utf8") ,unicode(row[16], "utf8") ,unicode(row[17], "utf8"),
   #]
     # cur.execute("INSERT INTO workloads_conf (spark_executor_memory,spark_shuffle_compress,spark_rdd_compress,spark_executor_cores,spark_memory_storageFraction,spark_memory_fraction,spark_executor_instances,spark_broadcat_blockSize,spark_default_parallelism,spark_memory_offHeap_size,spark_shuffle_file_buffer,spark_shuffle_spill_compress,spark_shuffle_spill,spark_memory_offHeap_enabled,spark_speculation,spark_app_name,app_id,execution_time) VALUES (?, ?, ?,?,? , ?, ?, ? , ? ,? ,?,?,?,?,?,?,?,?);", to_db)

    con.commit()
    con.close()


def load_data():
    
#    csv_path = '/local/data/experiment/bayes/bayes_samples_30params.csv'
    csv_path = '/local/data/experiment/tpch_samples.csv'
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    config = Configuration()
    reader = csv.reader(open(csv_path,'r'), delimiter=',')
    next(reader, None) #skip header
    for row in reader:

#        print (">>>> " , row )
        config.spark_executor_cores= int (row [0])
        config.spark_executor_memory= int (row [1])
        config.spark_executor_instances= int (row [2])
        config.spark_default_map_parallelism= int (row [3])
        config.spark_memory_offHeap_enabled= row [4]
        config.spark_memory_offHeap_size= int (row [5])

        config.spark_memory_fraction= float (row [6])
        config.spark_memory_storageFraction= float (row [7])
        config.spark_shuffle_file_buffer= int (row [8])
        config.spark_speculation= row [9]
        config.spark_reducer_maxSizeInFlight=int( row [10])
        config.spark_shuffle_sort_bypassMergeThreshold= int (row [11])
        config.spark_speculation_interval= int (row [12])
        config.spark_speculation_multiplier= float (row [13])
        config.spark_speculation_quantile= float (row [14])
        config.spark_broadcast_blockSize= int(row [15])
        config.spark_io_compression_codec= row [16]
        config.spark_io_compression_lz4_blockSize= int (row [17])
        config.spark_io_compression_snappy_blockSize= int(row [18])
        config.spark_kryo_referenceTracking= row [19]
        config.spark_kryoserializer_buffer_max= int (row [20])
        config.spark_kryoserializer_buffer= int (row [21])
        config.spark_storage_memoryMapThreshold= int (row [22])

        config.spark_network_timeout= int (row [23])
        config.spark_locality_wait= int (row [24])
        config.spark_shuffle_compress= row [25]
        config.spark_shuffle_spil_compress= row [26]
        config.spark_broadcast_compress= row [27]
        config.spark_rdd_compress= row [28]
        config.spark_serializer= row [29]
        config.execution_time = int (row [30])
        config.spark_app_name =  row [31]
        config.spark_app_id = row [32]
        config.input_size = row [33]

        #config.spark_executor_cores = row [0]
        #config.spark_executor_memory = row [1]
        #config.spark_shuffle_sort_bypassMergeThreshold=row [2]
        #config.spark_io_compression_lz4_blockSize = row [3]
        
    
        to_db =   [config.spark_executor_memory , unicode(str(config.spark_shuffle_compress).upper(), "utf8") ,  unicode( str(config.spark_rdd_compress).upper(), "utf8"), config.spark_executor_cores ,  float("{0:.2f}".format(config.spark_memory_storageFraction )) ,  float("{0:.2f}".format(config.spark_memory_fraction )) , config.spark_executor_instances , config.spark_broadcat_blockSize , config.spark_default_parallelism ,  config.spark_memory_offHeap_size, config. spark_shuffle_file_buffer  ,  unicode( str(config.spark_shuffle_spill_compress).upper() , "utf8")  , unicode(str(config.spark_memory_offHeap_enabled).upper() , "utf8"),unicode(str(config.spark_speculation).upper() , "utf8") , config.spark_reducer_maxSizeInFlight, config.spark_shuffle_sort_bypassMergeThreshold , config.spark_speculation_interval , config.spark_io_compression_lz4_blockSize, config.spark_io_compression_snappy_blockSize , config.spark_kryoserializer_buffer_max , config.spark_kryoserializer_buffer , config.spark_storage_memoryMapThreshold, config.spark_network_timeout, config.spark_locality_wait ,  float("{0:.2f}".format(config.spark_speculation_multiplier)), float("{0:.2f}".format(config.spark_speculation_quantile )) , config.spark_kryo_referenceTracking  ,  unicode(str(config.spark_broadcast_compress).upper() , "utf8")   , unicode(str(config. spark_io_compression_codec ), "utf8"),  unicode(str(config. spark_serializer ), "utf8")  ,  config.execution_time , unicode(config.spark_app_name , "utf8") , unicode(config.spark_app_id , "utf8") , config.input_size]
        
        config.spark_memory_storageFraction = float("{0:.2f}".format(config.spark_memory_storageFraction ))
        config.spark_memory_fraction = float("{0:.2f}".format(config.spark_memory_fraction ))
        config.spark_speculation_multiplier = float("{0:.2f}".format(config.spark_speculation_multiplier ))
        config.spark_speculation_quantile = float("{0:.2f}".format(config.spark_speculation_quantile ))
        cur.execute("INSERT INTO workloads_30conf (spark_executor_memory,spark_shuffle_compress,spark_rdd_compress,spark_executor_cores,spark_memory_storageFraction,spark_memory_fraction,spark_executor_instances,spark_broadcat_blockSize,spark_default_parallelism,spark_memory_offHeap_size,spark_shuffle_file_buffer,spark_shuffle_spill_compress,spark_memory_offHeap_enabled,spark_speculation,spark_reducer_maxSizeInFlight, spark_shuffle_sort_bypassMergeThreshold , spark_speculation_interval , spark_io_compression_lz4_blockSize, spark_io_compression_snappy_blockSize , spark_kryoserializer_buffer_max , spark_kryoserializer_buffer , spark_storage_memoryMapThreshold, spark_network_timeout, spark_locality_wait ,  spark_speculation_multiplier , spark_speculation_quantile , spark_kryo_referenceTracking , spark_broadcast_compress ,  spark_io_compression_codec , spark_serializer , execution_time , spark_app_name, spark_app_id , input_size) VALUES (?, ?, ?,?,? , ? ,? ,?,?,?,?,?,?,?,? ,?,?,?,?,? ,?,?,?,?,? ,?,?,?,?,?,?,?,?, ?);", to_db)

    con.commit()
    con.close()




def  insert_conf (config , cost):
    print (">>>>>>> inserting conf and cost into DB ......")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    
    to_db =   [config.spark_executor_memory , unicode(str(config.spark_shuffle_compress).upper(), "utf8") ,  unicode( str(config.spark_rdd_compress).upper(), "utf8"), config.spark_executor_cores ,  float("{0:.2f}".format(config.spark_memory_storageFraction )) ,  float("{0:.2f}".format(config.spark_memory_fraction )) , config.spark_executor_instances , config.spark_broadcat_blockSize , config.spark_default_parallelism ,  config.spark_memory_offHeap_size, config. spark_shuffle_file_buffer  ,  unicode( str(config.spark_shuffle_spill_compress).upper() , "utf8")  , unicode(str(config.spark_memory_offHeap_enabled).upper() , "utf8"), unicode(str(config.spark_speculation).upper() , "utf8") , config.spark_reducer_maxSizeInFlight, config.spark_shuffle_sort_bypassMergeThreshold , config.spark_speculation_interval , config.spark_io_compression_lz4_blockSize, config.spark_io_compression_snappy_blockSize , config.spark_kryoserializer_buffer_max , config.spark_kryoserializer_buffer , config.spark_storage_memoryMapThreshold, config.spark_network_timeout, config.spark_locality_wait ,  float("{0:.2f}".format(config.spark_speculation_multiplier)) , float("{0:.2f}".format(config.spark_speculation_quantile )) , config.spark_kryo_referenceTracking  ,  unicode(str(config.spark_broadcast_compress).upper() , "utf8")   , unicode(str(config. spark_io_compression_codec ), "utf8") ,  unicode(str(config. spark_serializer ), "utf8")  ,  cost, unicode(config.spark_app_name , "utf8") , unicode(config.spark_app_id , "utf8") , config.input_size]
    

    config.spark_memory_storageFraction = float("{0:.2f}".format(config.spark_memory_storageFraction ))
    config.spark_memory_fraction = float("{0:.2f}".format(config.spark_memory_fraction ))
    config.spark_speculation_multiplier = float("{0:.2f}".format(config.spark_speculation_multiplier ))
    config.spark_speculation_quantile = float("{0:.2f}".format(config.spark_speculation_quantile ))

    cur.execute("INSERT INTO workloads_30conf (spark_executor_memory,spark_shuffle_compress,spark_rdd_compress,spark_executor_cores,spark_memory_storageFraction,spark_memory_fraction,spark_executor_instances,spark_broadcat_blockSize,spark_default_parallelism,spark_memory_offHeap_size,spark_shuffle_file_buffer,spark_shuffle_spill_compress,spark_memory_offHeap_enabled,spark_speculation,spark_reducer_maxSizeInFlight, spark_shuffle_sort_bypassMergeThreshold , spark_speculation_interval , spark_io_compression_lz4_blockSize, spark_io_compression_snappy_blockSize , spark_kryoserializer_buffer_max , spark_kryoserializer_buffer , spark_storage_memoryMapThreshold, spark_network_timeout, spark_locality_wait ,  spark_speculation_multiplier , spark_speculation_quantile , spark_kryo_referenceTracking , spark_broadcast_compress ,  spark_io_compression_codec , spark_serializer , execution_time , spark_app_name, spark_app_id , input_size) VALUES (?, ?, ?,?,? , ? ,? ,?,?,?,?,?,?,?, ? ,?,?,?,?,? ,?,?,?,?,? ,?,?,?,?,?,?,?,?, ?);", to_db)
    con.commit()
    con.close()

def remove_noise ():
    print (">>>>>>> removing noisy conf from  DB ......")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    res = cur.execute("delete from workloads_30conf  where  execution_time < 200000")
    rows = res.fetchall()
    print(">>>>n_rows >>> " , str(len(rows))) 
#    cur.execute("delete from workloads_30conf  where spark_app_id LIKE 'application_1547014108029_0001' ")
 #   cur.execute("delete from workloads_30conf  where spark_app_id LIKE 'application_1547014108029_0002' ")
  #  cur.execute("delete from workloads_30conf  where spark_app_id LIKE 'application_1547014108029_0003' ")
   # cur.execute("delete from workloads_30conf  where spark_app_id LIKE 'application_1547014108029_0004' ")
   # cur.execute("delete from workloads_30conf  where spark_app_id LIKE 'application_1547014108029_0005' ")
    con.commit()
    con.close()

def empty_table ():
    print (">>>>>>> empty  DB ......")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("delete from workloads_30conf")
    con.commit()
    con.close()


def execute_query ():
    print (">>>>>>> executing query ......")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    res = cur.execute("select * from workloads_30conf  where  spark_app_name LIKE 'pagerank' ")
    rows = res.fetchall()
    print (" >>>> " , len (rows))
    print (">>>>" , str (rows [0]))
    con.commit()
    con.close()

def getCost(config):

    print (">>>>>>>>>>>>>>>>> getting the cost form  DB ....")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print (">>>> get cost of  >>>> core >>> " , config.spark_executor_cores)
    print (">>>> get cost of >>>>> memory >>> " , config.spark_executor_memory)
    query = "select  execution_time from workloads_30conf  where spark_executor_memory= ? AND spark_shuffle_compress = ? AND spark_rdd_compress =? AND spark_executor_cores = ? AND spark_memory_storageFraction = ? AND spark_memory_fraction = ? AND spark_executor_instances = ? AND spark_broadcat_blockSize = ? AND spark_default_parallelism = ? AND spark_memory_offHeap_size = ? AND spark_shuffle_file_buffer = ? AND spark_shuffle_spill_compress=  ?  AND spark_memory_offHeap_enabled = ? AND spark_speculation = ? AND spark_reducer_maxSizeInFlight= ? AND  spark_shuffle_sort_bypassMergeThreshold = ? AND  spark_speculation_interval = ? AND  spark_io_compression_lz4_blockSize= ? AND  spark_io_compression_snappy_blockSize = ? AND  spark_kryoserializer_buffer_max = ? AND  spark_kryoserializer_buffer = ? AND  spark_storage_memoryMapThreshold= ?  AND  spark_network_timeout= ? AND  spark_locality_wait = ? AND    spark_speculation_multiplier = ? AND  spark_speculation_quantile = ? AND  spark_kryo_referenceTracking = ? AND   spark_broadcast_compress = ?  AND   spark_io_compression_codec = ? AND  spark_serializer = ?  AND  spark_app_name = ? AND input_size =?"
    


#    result= cur.execute ( query ,  ( config.spark_executor_memory , unicode(config.spark_shuffle_compress, "utf8") ,  unicode(config.spark_rdd_compress, "utf8"), config.spark_executor_cores ,  config.spark_memory_storageFraction,  config.spark_memory_fraction, config.spark_executor_instances , config.spark_broadcat_blockSize , config.spark_default_parallelism ,  config.spark_memory_offHeap_size, config. spark_shuffle_file_buffer  ,  unicode( config.spark_shuffle_spill_compress , "utf8"),  unicode(config.spark_shuffle_spill , "utf8")  , unicode(config.spark_memory_offHeap_enabled , "utf8"), unicode(config.spark_speculation , "utf8") , unicode(config.spark_app_name , "utf8")))
    config.spark_memory_storageFraction = float("{0:.2f}".format(config.spark_memory_storageFraction ))

    config.spark_memory_fraction = float("{0:.2f}".format(config.spark_memory_fraction ))
    config.spark_speculation_multiplier = float("{0:.2f}".format(config.spark_speculation_multiplier ))
    config.spark_speculation_quantile = float("{0:.2f}".format(config.spark_speculation_quantile ))
    
    result= cur.execute ( query ,  ( config.spark_executor_memory , unicode(str(config.spark_shuffle_compress).upper(), "utf8") ,  unicode( str(config.spark_rdd_compress).upper(), "utf8"), config.spark_executor_cores ,  config.spark_memory_storageFraction,  config.spark_memory_fraction, config.spark_executor_instances , config.spark_broadcat_blockSize , config.spark_default_parallelism ,  config.spark_memory_offHeap_size, config. spark_shuffle_file_buffer  ,  unicode( str(config.spark_shuffle_spill_compress).upper() , "utf8") , unicode(str(config.spark_memory_offHeap_enabled).upper() , "utf8"), unicode(str(config.spark_speculation).upper() , "utf8"),config.spark_reducer_maxSizeInFlight, config.spark_shuffle_sort_bypassMergeThreshold , config.spark_speculation_interval , config.spark_io_compression_lz4_blockSize, config.spark_io_compression_snappy_blockSize , config.spark_kryoserializer_buffer_max , config.spark_kryoserializer_buffer , config.spark_storage_memoryMapThreshold , config.spark_network_timeout, config.spark_locality_wait , config.spark_speculation_multiplier , config.spark_speculation_quantile , config.spark_kryo_referenceTracking   ,  unicode(str(config.spark_broadcast_compress).upper() , "utf8")   , unicode(str(config. spark_io_compression_codec ), "utf8") ,  unicode(str(config. spark_serializer ), "utf8") , unicode(config.spark_app_name , "utf8") , config.input_size))

    #result = cur.execute ("select * from workloads_conf")
    print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_compress).upper(), "utf8"))  
    print (">>>>>>>>>>> " , unicode( str(config.spark_rdd_compress).upper(), "utf8"))
    print (">>>>>>>>>>> " , str (config.spark_memory_storageFraction))
    print (">>>>>>>>>>> " , str(config.spark_memory_fraction))
    print (">>>>>>>>>>> " , str (config.spark_executor_instances))
    print (">>>>>>>>>>> " , str(config.spark_broadcat_blockSize))
    print (">>>>>>>>>>> " ,str(config.spark_default_parallelism))
    print (">>>>>>>>>>> " , str(config.spark_memory_offHeap_size))
    print (">>>>>>>>>>> " , str(config.spark_shuffle_file_buffer))
    print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_spill_compress).upper(), "utf8"))
    #print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_spill).upper(), "utf8"))
    print (">>>>>>>>>>> " , unicode(str(config.spark_memory_offHeap_enabled).upper(), "utf8"))
    print (">>>>>>>>>>> " , unicode(str(config.spark_speculation).upper(), "utf8"))
    print (">>>>>>>>>>> " , unicode(str(config.spark_app_name), "utf8"))
    
    
    
    
    
    
    
    
    
#    print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_compress).upper(), "utf8"))
 #   print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_compress).upper(), "utf8"))
  #   print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_compress).upper(), "utf8"))
   





   # cost = result.fetchone()
    rows = result.fetchall()
    print ("row count >>>  " , len(rows))
    if len(rows) == 0:
       return 1000000000
    cost = rows[0]
    print ( " >>>> cost >>> " , cost)
    print  int(cost[len(cost) -2])
    con.commit()
    con.close()
    return int(cost[len(cost) -2])   #cost index is before last
    return cost


def getAppId(config):

    print (">>>>>>>>>>>>>>>>> getting the cost form  DB ....")
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    print (">>>> get cost of  >>>> core >>> " , config.spark_executor_cores)
    print (">>>> get cost of >>>>> memory >>> " , config.spark_executor_memory)
    query = "select  spark_app_id  from workloads_30conf  where spark_executor_memory= ? AND spark_shuffle_compress = ? AND spark_rdd_compress =? AND spark_executor_cores = ? AND spark_memory_storageFraction = ? AND spark_memory_fraction = ? AND spark_executor_instances = ? AND spark_broadcat_blockSize = ? AND spark_default_parallelism = ? AND spark_memory_offHeap_size = ? AND spark_shuffle_file_buffer = ? AND spark_shuffle_spill_compress=  ?  AND spark_memory_offHeap_enabled = ? AND spark_speculation = ? AND spark_reducer_maxSizeInFlight= ? AND  spark_shuffle_sort_bypassMergeThreshold = ? AND  spark_speculation_interval = ? AND  spark_io_compression_lz4_blockSize= ? AND  spark_io_compression_snappy_blockSize = ? AND  spark_kryoserializer_buffer_max = ? AND  spark_kryoserializer_buffer = ? AND  spark_storage_memoryMapThreshold= ?  AND  spark_network_timeout= ? AND  spark_locality_wait = ? AND   spark_speculation_multiplier = ? AND  spark_speculation_quantile = ? AND  spark_kryo_referenceTracking = ? AND     spark_broadcast_compress = ?  AND   spark_io_compression_codec = ? AND  spark_serializer = ?  AND  spark_app_name = ?  AND input_size =?"
    


#    result= cur.execute ( query ,  ( config.spark_executor_memory , unicode(config.spark_shuffle_compress, "utf8") ,  unicode(config.spark_rdd_compress, "utf8"), config.spark_executor_cores ,  config.spark_memory_storageFraction,  config.spark_memory_fraction, config.spark_executor_instances , config.spark_broadcat_blockSize , config.spark_default_parallelism ,  config.spark_memory_offHeap_size, config. spark_shuffle_file_buffer  ,  unicode( config.spark_shuffle_spill_compress , "utf8"),  unicode(config.spark_shuffle_spill , "utf8")  , unicode(config.spark_memory_offHeap_enabled , "utf8"), unicode(config.spark_speculation , "utf8") , unicode(config.spark_app_name , "utf8")))
    config.spark_memory_storageFraction = float("{0:.2f}".format(config.spark_memory_storageFraction ))

    config.spark_memory_fraction = float("{0:.2f}".format(config.spark_memory_fraction ))
    config.spark_speculation_multiplier = float("{0:.2f}".format(config.spark_speculation_multiplier ))
    config.spark_speculation_quantile = float("{0:.2f}".format(config.spark_speculation_quantile ))
    
    result= cur.execute ( query ,  ( config.spark_executor_memory , unicode(str(config.spark_shuffle_compress).upper(), "utf8") ,  unicode( str(config.spark_rdd_compress).upper(), "utf8"), config.spark_executor_cores ,  config.spark_memory_storageFraction,  config.spark_memory_fraction, config.spark_executor_instances , config.spark_broadcat_blockSize , config.spark_default_parallelism ,  config.spark_memory_offHeap_size, config. spark_shuffle_file_buffer  ,  unicode( str(config.spark_shuffle_spill_compress).upper() , "utf8") , unicode(str(config.spark_memory_offHeap_enabled).upper() , "utf8"), unicode(str(config.spark_speculation).upper() , "utf8"),config.spark_reducer_maxSizeInFlight, config.spark_shuffle_sort_bypassMergeThreshold , config.spark_speculation_interval , config.spark_io_compression_lz4_blockSize, config.spark_io_compression_snappy_blockSize , config.spark_kryoserializer_buffer_max , config.spark_kryoserializer_buffer , config.spark_storage_memoryMapThreshold,  config.spark_network_timeout, config.spark_locality_wait , config.spark_speculation_multiplier , config.spark_speculation_quantile , config.spark_kryo_referenceTracking   ,  unicode(str(config.spark_broadcast_compress).upper() , "utf8")  , unicode(str(config. spark_io_compression_codec ), "utf8") ,  unicode(str(config. spark_serializer ), "utf8") , unicode(config.spark_app_name , "utf8") , config.input_size))

    #result = cur.execute ("select * from workloads_conf")
    print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_compress).upper(), "utf8"))  
    print (">>>>>>>>>>> " , unicode( str(config.spark_rdd_compress).upper(), "utf8"))
    print (">>>>>>>>>>> " , str (config.spark_memory_storageFraction))
    print (">>>>>>>>>>> " , str(config.spark_memory_fraction))
    print (">>>>>>>>>>> " , str (config.spark_executor_instances))
    print (">>>>>>>>>>> " , str(config.spark_broadcat_blockSize))
    print (">>>>>>>>>>> " ,str(config.spark_default_parallelism))
    print (">>>>>>>>>>> " , str(config.spark_memory_offHeap_size))
    print (">>>>>>>>>>> " , str(config.spark_shuffle_file_buffer))
    print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_spill_compress).upper(), "utf8"))
    #print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_spill).upper(), "utf8"))
    print (">>>>>>>>>>> " , unicode(str(config.spark_memory_offHeap_enabled).upper(), "utf8"))
    print (">>>>>>>>>>> " , unicode(str(config.spark_speculation).upper(), "utf8"))
    print (">>>>>>>>>>> " , unicode(str(config.spark_app_name), "utf8"))
    
#    print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_compress).upper(), "utf8"))
 #   print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_compress).upper(), "utf8"))
  #   print (">>>>>>>>>>> " , unicode(str(config.spark_shuffle_compress).upper(), "utf8"))
   
   # cost = result.fetchone()
    rows = result.fetchall()
    print ("row count >>>  " , len(rows))
    if len(rows) == 0:
       return 1000000000
    id = rows[0]
    print ( " >>>> id >>> " , id)
    print  id [len(id) -2]
    con.commit()
    con.close()
    return id [len(id) -2]   #cost index is before last
    return id



class Configuration:
    count = 0
    spark_executor_memory = 1
    spark_shuffle_compress= False
    spark_rdd_compress=False
    spark_executor_cores = 16
    spark_memory_storageFraction= 0.5
    spark_memory_fraction= 0.6
    spark_executor_instances= 5
    spark_broadcat_blockSize=4   ### in MB
    spark_default_parallelism=8 
    spark_memory_offHeap_size =0 
    spark_shuffle_file_buffer=32
    spark_shuffle_spill_compress = True
    #spark_shuffle_spill = True
    spark_memory_offHeap_enabled= False
    spark_speculation = False
    spark_app_name = ""
    spark_app_id = "#"
    input_size = 0
    spark_reducer_maxSizeInFlight= 48       # in MB
    spark_shuffle_sort_bypassMergeThreshold =  200
    spark_speculation_interval = 100
    spark_io_compression_lz4_blockSize = 32
    spark_io_compression_snappy_blockSize = 32
    spark_kryoserializer_buffer_max = 64 # in MB
    spark_kryoserializer_buffer =64
    spark_storage_memoryMapThreshold=2 #in MB
    #spark_akka_failure_detector_threshold =  300
    #spark_akka_heartbeat_pauses = 6000
    #spark_akka_heartbeat_interval= 1000
    #spark_akka_threads =4
    spark_network_timeout= 120
    spark_locality_wait= 3
    #spark_scheduler_revive_interval= 1
    #spark_task_maxFailures =4
    spark_speculation_multiplier =1.5
    spark_speculation_quantile =0.75
    spark_kryo_referenceTracking =True
    #spark_shuffle_consolidateFiles =False
    spark_broadcast_compress = True 
   # spark_localExecution_enabled =False    
    spark_io_compression_codec = "lz4" 
    spark_serializer = "org.apache.spark.serializer.JavaSerializer"
    #spark_shuffle_manager =  "sort"
    #spark_file_fetchTimeout = 60
    
    
    
    
    
    
    
    
    

    def __init__ (self):
        print (">>>>>>>>>>>>>>>>>>>> creating conf object" )


################ testing ###############################
#create_database()
conf = Configuration()
conf.spark_app_name = "ScalaPageRank"

conf.spark_executor_memory = 2
conf.spark_shuffle_compress= True
conf.spark_rdd_compress=False
conf.spark_executor_cores = 9
conf.spark_memory_storageFraction= 0.8
conf.spark_memory_fraction= 0.54
conf.spark_executor_instances= 12
conf.spark_broadcat_blockSize=39385
conf.spark_default_parallelism=22
conf.spark_memory_offHeap_size =39845
conf.spark_shuffle_file_buffer=64
conf.spark_shuffle_spill_compress = False
#conf.spark_shuffle_spill = True
conf.spark_memory_offHeap_enabled= True
conf.spark_speculation = False


remove_noise()

##cost = getCost (conf) # get the default conf cost
##print (">>>cost >> " , str(cost))

#create_table()
#empty_table ()

#execute_query()
#load_data()

