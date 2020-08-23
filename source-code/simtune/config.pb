language: PYTHON
name: "sparkTuner_offline_hibench_sig_params_dynamic"


# Task indicator

variable {
 name: "Task"
 type: INT
 size: 1
 min:  0
 max:  5
}


variable {
   name: "spark.speculation.quantile"
   type:FLOAT
   size: 1
   min: 0
   max:1
}

variable {
   name: "spark.executor.memory"
   type:INT
   size: 1
   min: 5
   max:43
}

variable {
   name: "hibench.default.map.parallelism"
   type:INT
   size: 1
   min: 8
   max:50
}

variable {
   name: "Time.spark.locality.wait"
   type:INT
   size: 1
   min:1
   max:10
}
variable {
   name: "MB.spark.storage.memoryMapThreshold"
   type:INT
   size: 1
   min:50
   max:500
}

variable {
   name: "spark.executor.instances"
   type:INT
   size: 1
   min: 8
   max: 48
}
