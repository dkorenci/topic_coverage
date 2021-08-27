#!/usr/bin/env bash
# copy all resources necessary for running an experiment to the server

ssh-agent
ssh-add ~/.ssh/dam1root
server=$1
folder=$2
scp run_experiment.sh "$server:$folder" 
scp master_run.sh "$server:$folder" 
scp lda_model_img.tar "$server:$folder" 
ssh-add -D
