
**Prerequiste:** make sure that speaarmint framwork works propoerly through running braninpy example as described in https://github.com/JasperSnoek/spearmint

To run SimTune example:
* cd code
* edit simtune\config.pb to point to the source workload (i.e. set the 'name' parameter to the file that contains the workload code) and the configuration paramters then start tuning using the following command:

```
python spearmint_sync.py simtune --method=MultiTaskEIOptChooser --method-args=\"task_num=0\"  --max-finished-jobs=15
```
* To transfer this tuning knowledge over to a target workload: edit simtune\config.pb to point to the target workload then start tuning using the following command:
```
python spearmint_sync.py simtune --method=MultiTaskEIOptChooser --method-args=\"task_num=1\"  --max-finished-jobs=30
```

