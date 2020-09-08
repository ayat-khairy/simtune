
**Prerequiste:** make sure that Keras is installed and speaarmint framwork works propoerly through running braninpy example as described in https://github.com/JasperSnoek/spearmint

To run Similarity Analysis example:
1. build the workload representation model:
```
cd source-code/similarity-analysis
python exec_features_representation_learner.py 
```
2. run example of finding a similar workload to Bayes-BD (one of HiBench workloads https://github.com/Intel-bigdata/HiBench):
```
cd source-code/similarity-analysis
python analyze_similarity.py
```

To run SimTune example:
1. cd source-code
2. edit simtune\config.pb to point to the source workload (i.e. set the 'name' parameter to the file that contains the workload code) and the configuration paramters then start tuning using the following command:

```
python spearmint_sync.py simtune --method=MultiTaskEIOptChooser --method-args=\"task_num=0\"  --max-finished-jobs=15
```
3. To transfer this tuning knowledge over to a target workload: edit simtune\config.pb to point to the target workload then start tuning using the following command:
```
python spearmint_sync.py simtune --method=MultiTaskEIOptChooser --method-args=\"task_num=1\"  --max-finished-jobs=30
```

