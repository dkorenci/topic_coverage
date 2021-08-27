#!/usr/bin/env bash

# run experiment (localy) from docker 'base' image - image containing the plaform
# the code and resources are local folders that are mounted into the container

code="/data/code/topic_coverage
      /data/code/pytopia_contexts/gtar_context /data/code/pytopia_contexts/phenotype_context
      /data/code/pyutils /data/code/pytopica /data/code/pymedialab /data/code/pyldazen
      /data/code/hca/HCA-0.63"
data="/data/resources/ /datafast/topic_coverage/"

IMAGE=topic_coverage

# add source folders to mount options and pythonpath
mount=""; pypath=""
for f in $code; do
    mount="$mount --mount type=bind,src=$f,dst=$f"
    pypath="$pypath:$f"    
done
# add data folders to mount options
for f in $data; do
    mount="$mount --mount type=bind,src=$f,dst=$f"
done
# setup env variables
env="--env PYTHONPATH=$pypath"

# user and group to which ownership of files created/modified 
# within the container (with root account) will be restored
origuser=damir 
waytorun=$1
if [ $waytorun == "-i" ]; then 
    # execute the container interactively
    docker run -t -i $mount $env $IMAGE
else # execute python script specified by parameters    
    script=$1 # full path to py script to be executed
    runset=$2 # identifier of a set of params of models to be built
    # folder where models will be stored, has to be mounted
    buildFolder=$3
    mount="$mount --mount type=bind,src=$buildFolder,dst=$buildFolder"
    # CREATE ENTRYPOINT SCRIPT THAT WILL BE EXECUTED WITHIN THE CONTAINER
    entrypoint=entrypoint.sh
    echo "useradd damir" >> $entrypoint
    echo "echo DELETING PYC" >> $entrypoint
    for f in $code; do
        echo "find $f -name \"*.pyc\" -delete" >> $entrypoint
    done
    runPath=$(dirname "$script"); scriptFile=$(basename "$script")
    # script running code
    echo "echo RUNNING BUILD SCRIPT" >> $entrypoint
    echo "cd $runPath" >> $entrypoint
    echo "python $scriptFile $runset $buildFolder" >> $entrypoint
    # clear compiled files
    echo "echo DELETING PYC" >> $entrypoint
    for f in $code; do
        echo "find $f -name \"*.pyc\" -delete" >> $entrypoint
    done
    # set folder permissions back to local user
    reclaim="$runPath $buildFolder"
    echo "echo RECLAIMING FOLDER OWNERSHIP" >> $entrypoint
    for f in $reclaim; do
        echo "chown -R damir:damir $f" >> $entrypoint
    done
    chmod +x $entrypoint
    
    # EXECUTE CONTAINER
    mount="$mount --mount type=bind,src=`pwd`,dst=/entrypoint"
    runCmd="/bin/bash /entrypoint/$entrypoint"
    #echo $mount
    echo EXECUTING CONTAINER
    docker run -a stdout -a stderr $mount $env $IMAGE $runCmd
    
    # cleanup
    rm $entrypoint # remove entrypoint script
fi
