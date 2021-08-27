#!/usr/bin/env bash

# run experiment from docker 'run' image - image containing code and resources

waytorun=$1

#### execute the container interactively
if [ $waytorun == "-i" ]; then
    docker run -t -i $2
    exit 0
fi

#### execute a python script inside the container

echo "STARTING RUN"
runimage=$1
script=$2 # full path to py script to be executed
paramset=$3 # identifier of a set of params of models to be built
paramsetSplits=$4
paramsetPart=$5
modelFolder=$6 # folder where models will be stored, has to be mounted

# setup run folder (timestamped, rnd. string stamped)
timestamp=`date +"%Y.%m.%d-%H.%M.%S"`
rndstamp=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 10`
runid="$runimage[$timestamp][$rndstamp]"
echo "RUN FOLDER: $runid"
[ -d $runid ] && rm -rf $runid
mkdir $runid

# CREATE ENTRYPOINT SCRIPT THAT WILL BE EXECUTED WITHIN THE CONTAINER
runfolder="`pwd`/$runid"
entrypoint="$runfolder/entrypoint.sh"
mount="$mount -v $runfolder:/runfolder"
runfolderC="/runfolder" # container mountpoint for runfolder
runCmd="/bin/bash $runfolderC/$(basename $entrypoint)"
# mount model folder into the container
mount="$mount -v $modelFolder:$modelFolder"
# python script running code
runPath=$(dirname "$script"); scriptFile=$(basename "$script")
echo "echo RUNNING BUILD SCRIPT" >> $entrypoint
echo "cd $runPath" >> $entrypoint
echo "export OPENBLAS_NUM_THREADS=1" >> $entrypoint
echo "python -OO -u $scriptFile $paramset $paramsetSplits $paramsetPart $modelFolder >> $scriptFile.out" >> $entrypoint
# copy python logs and diagnostics
echo "cp $scriptFile.out $runfolderC" >> $entrypoint
echo "cp -r pylog $runfolderC" >> $entrypoint
# set some created folder and files ownership back to local user
userid="$UID"
if [ "$userid" != "" ]; then
    echo "useradd user -u $userid" >> $entrypoint
    reclaim="$runfolderC"
    echo "echo RECLAIMING FOLDER OWNERSHIP" >> $entrypoint
    for f in $reclaim; do
        echo "chown -R user:user $f" >> $entrypoint
    done
fi

chmod +x $entrypoint

# EXECUTE ENTRYPOINT
echo EXECUTING ENTRYPOINT SCRIPT
echo "docker run $mount $runimage $runCmd"
docker run $mount $runimage $runCmd
#docker run -a stdout -a stderr $mount $IMAGE $runCmd

# COPY LOG AND DIAGNOSTICS
