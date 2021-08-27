#!/usr/bin/env bash

runimage=lda_model_stabil
script=/data/code/topic_coverage/topic_coverage/experiments/stability/experiment_runner.pyo
### SCRIPT PARAMS ###
modelFolder=/data/modelbuild/topic_coverage/docker_modelbuild/numt_prodbuild_sample/
covCache=/datafast/topic_coverage/numt_prod/function_cache_djurdja_working/
stabilCache=/datafast/topic_coverage/stabil_test/function_cache/stability/
func1=$1
func2=$2

for c in uspol pheno; do
    for m in lda alda nmf pyp; do
        echo "RUNNING $c $m"
        ./run_experiment_stabil.sh $runimage $script $modelFolder $covCache $stabilCache $c $func1 $func2 $m 2>&1 > run_experiment_stabil.$func1.$func2.$c.$m.out.txt
    done
done
