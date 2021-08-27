#!/usr/bin/env bash
# execute python modelbuild scripts on a local machine

export OPENBLAS_NUM_THREADS=1 # to ensure one thread per child script

script=$1 # full path to py script to be executed
paramset=$2 # identifier of a set of params of models to be built
paramsetParts=$3 # number of paramset parts
modelFolder=$4 # folder where models will be stored, has to be mounted

# setup PYTHONPATH
code=`cat code_folders.txt`
for f in $code; do
    PYTHONPATH="$PYTHONPATH:$f"    
done
export PYTHONPATH=$PYTHONPATH

for i in $(seq 0 $(expr $paramsetParts - 1)); do    
    python2 -u $script $paramset $paramsetParts $i $modelFolder 2>&1 > output$i.txt &
done
