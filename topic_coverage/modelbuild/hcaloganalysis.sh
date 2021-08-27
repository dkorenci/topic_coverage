#!/usr/bin/env bash

logfile=$1
fnamenoext="${logfile%%.*}"

# extract exp.topic entropy and perplexity
# COMMAND provides data about two phases: gibbs sampling, matrix estim.
#filter="perp|exp.ent|COMMAND"
filter="exp.ent|COMMAND"
cat $logfile | grep -E "$filter" | awk '{print $NF}' > "$fnamenoext.txt"
