#!/usr/bin/env bash
# run a set of docker builds of models defined with different parts of the same paramset

runimage=$1
script=$2 # full path to py script to be executed
paramset=$3 # identifier of a set of params of models to be built
paramsetParts=$4 # number of paramset parts
modelFolder=$5 # folder where models will be stored, has to be mounted

for i in $(seq 0 $(expr $paramsetParts - 1)); do
    ./run_experiment.sh $runimage $script $paramset $paramsetParts $i $modelFolder &
done