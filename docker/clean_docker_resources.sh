#!/usr/bin/env bash

#docker rm $(docker ps -a | grep "lda_model" | awk '{print $1}')

image=$1
cids=`\$\(docker ps -a | grep "$1" | awk '{print $1}'\)`
echo cids
#echo docker rm $(docker ps -a | grep \"$1\" | awk '{print $1}')
