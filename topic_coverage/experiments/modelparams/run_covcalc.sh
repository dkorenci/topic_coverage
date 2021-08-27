#!/usr/bin/env bash
export PYTHONPATH="/data/code/topic_coverage"
#exec=topic_coverage/experiments/modelparams/numt_calcplot_run.py
exec=numt_calcplot_run.py
python2.7 $exec \
    "/data/modelbuild/topic_coverage/docker_modelbuild/numt_prodbuild_sample/" \
    uspol sup.strict lda 2>&1 > run_covcalc.out.txt