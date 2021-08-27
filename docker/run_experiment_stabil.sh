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
### SCRIPT PARAMS ###
modelFolder=$3 # folder where models are stored, has to be mounted
covCache=$4 # folder for function's cache, has to be mounted
stabilCache=$5
corpus=$6
func1=$7
func2=$8
modeltyp=$9
### SCRIPT PARAMS ###

# setup run folder (timestamped, rnd. string stamped)
timestamp=`date +"%Y.%m.%d-%H.%M.%S"`
rndstamp=`head /dev/urandom | tr -dc A-Za-z0-9 | head -c 10`
runid="$runimage[$timestamp][$rndstamp][$corpus-$func1-$func2-$modeltyp]"
echo "RUN FOLDER: $runid"
[ -d $runid ] && rm -rf $runid
mkdir $runid

# CREATE ENTRYPOINT SCRIPT THAT WILL BE EXECUTED WITHIN THE CONTAINER
runfolder="`pwd`/$runid"
entrypoint="$runfolder/entrypoint.sh"
mount="$mount -v $runfolder:/runfolder"
runfolderC="/runfolder" # container mountpoint for runfolder
runCmd="/bin/bash $runfolderC/$(basename $entrypoint)"
# mount model and cache folders into the container
mount="$mount -v $modelFolder:$modelFolder"
mount="$mount -v $covCache:$covCache"
mount="$mount -v $stabilCache:$stabilCache"
# python script running code
runPath=$(dirname "$script"); scriptFile=$(basename "$script")
echo "echo RUNNING NUMT CALCULATION SCRIPT" >> $entrypoint
echo "cd $runPath" >> $entrypoint
echo "export OPENBLAS_NUM_THREADS=1" >> $entrypoint
echo "python -OO -u $scriptFile $modelFolder $covCache $stabilCache $corpus $func1 $func2 $modeltyp > $scriptFile.out" >> $entrypoint
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
