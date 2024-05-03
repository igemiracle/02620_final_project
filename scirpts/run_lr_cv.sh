#!/bin/bash

# set base path
BASE_PATH="../clean_data_pca/"$1
mkdir -p $BASE_PATH/metrics

# loop 10 times for 10 fold cv
for fold in {1..10}
do
    echo "Running fold $fold"
    # file path
    TRAIN_FEATURES="$BASE_PATH/fold_$fold/train_features.csv"
    TRAIN_LABELS="$BASE_PATH/fold_$fold/train_labels.csv"
    VALID_FEATURES="$BASE_PATH/fold_$fold/valid_features.csv"
    VALID_LABELS="$BASE_PATH/fold_$fold/valid_labels.csv"
    METRICS_FILE="$BASE_PATH/metrics/fold_${fold}_metrics.csv"

    # call the python script
    python lr_final.py $TRAIN_FEATURES $TRAIN_LABELS $VALID_FEATURES $VALID_LABELS $METRICS_FILE 50 0.01
done

