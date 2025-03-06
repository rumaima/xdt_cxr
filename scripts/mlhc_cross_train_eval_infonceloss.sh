#!/bin/bash
loss="_infonceloss"
command1="bash scripts/mlhc${loss}.sh"
command2="bash scripts/mlhc_eval${loss}.sh"

datasets=("shenzhen_cxr" "montgomery_cxr" "pneumonia_guangzhou" "covid")

this_dataset="$1"  # First command line argument

if [ -z "$this_dataset" ]; then
    echo "Error: Please provide a dataset as an argument."
    exit 1
fi

dataset_exists=false
for ds in "${datasets[@]}"; do
    if [ "$ds" == "$this_dataset" ]; then
        dataset_exists=true
        break
    fi
done

if [ "$dataset_exists" == false ]; then
    echo "Error: Dataset '$this_dataset' not found in the list of datasets."
    exit 1
fi

$command1 $this_dataset

for dataset in "${datasets[@]}"; do
    if [ "$dataset" != "$this_dataset" ]; then
        $command2 $this_dataset $dataset
    fi
done
