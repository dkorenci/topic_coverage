#!/usr/bin/env bash

runimage=lda_model_numt
script=/data/code/topic_coverage/topic_coverage/experiments/modelparams/numt_calcplot_run.pyo
### SCRIPT PARAMS ###
modelFolder=/data/modelbuild/topic_coverage/docker_modelbuild/numt_prodbuild_sample_small/
cacheFolder=/datafast/topic_coverage/numt_test/function_cache/
covfunc=$1
#covfunc=sup.strict
#covfunc=ctc

#modeltyp=lda
#corpus=uspol
#./run_experiment_numt.sh $runimage $script $modelFolder $cacheFolder $corpus $covfunc $modeltyp 2>&1 > run_experiment_numt.out.txt &

for c in uspol pheno; do
    for m in lda alda nmf pyp; do
        echo "RUNNING $c $m"
        ./run_experiment_numt.sh $runimage $script $modelFolder $cacheFolder $c $covfunc $m 2>&1 > run_experiment_numt.$covfunc.$c.$m.out.txt
    done
done
