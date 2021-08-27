#!/usr/bin/env bash

docker build --no-cache -t topic_coverage --file topic_coverage.docker .
#docker build -t topic_coverage:initial --file topic_coverage.docker .

# create cont. cname from image
#sudo docker start cname
#sudo docker exec -ti cname
#/etc/profile.d/
#export PYTHONPATH=$PYTHONPATH:/code/pyutils/   

