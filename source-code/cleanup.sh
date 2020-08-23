#! /bin/bash
# This is a simple script to cleanup the intermediate files in 
# spearmint experiment directories
[[ -n "$1" ]] || { echo "Usage: cleanup.sh <experiment_dir>"; exit 0 ; }
if [ -d $1 ]
then
    cd $1
    rm trace.csv
    rm output/*
    rm jobs/*
    rm expt-grid.pkl
    rm expt-grid.pkl.lock
    rm *Chooser*.pkl
    rm GaussianProcessClassifier*.pkl
    rm GaussianProcessRegressor*.pkl
    rm *.lock
    rm *Chooser*hyperparameters.txt
    rm best_job_and_result.txt
    rm trace.json
    rm info.txt
    rm contours/*
else
    echo "$1 is not a valid directory"
fi
