#!/usr/bin/env bash

# build docker image based on base/platform image
# the image contains code and data resources necessary for running coverage experiments

#### Build setup

# image names
BASE_IMAGE=topic_coverage
EXEC_IMAGE=lda_model_numt
# ##############

# files with code and resource folders
# the container will be set up to contain the folders with the same paths
code=`cat code_folders.txt`
resources=`cat resource_folders.txt`
echo "BUILDING IMAGE $EXEC_IMAGE FROM $BASE_IMAGE"
# build folder and dockerfile setup
buildfolder="`pwd`/dockerbuild"
[ -d $buildfolder ] && sudo rm -rf $buildfolder
mkdir $buildfolder
echo "BUILD FOLDER:$buildfolder"
dockerfile="$buildfolder/Dockerfile"

#### Dockerfile creation

# base image
echo "FROM $BASE_IMAGE" > $dockerfile
# copy code folders to buildfolder and remove unnecessary files
# create ADD commands in dockerfile that copy these folders to image
# setup environment (pythonpath)
echo "ENV PYTHONPATH \"\"" >> $dockerfile
echo "COPYING AND CLEANING CODE"
for f in $code; do
    echo $f
    cp -r $f $buildfolder
    folder=`basename $f` # last folder in the path
    find "$buildfolder/$folder" -name "*.pyc" -delete
    find "$buildfolder/$folder" -name ".git" -prune -exec rm -rf "{}" \;
    find "$buildfolder/$folder" -name ".idea" -prune -exec rm -rf "{}" \;
    find "$buildfolder/$folder" -name "pylog" -prune -exec rm -rf "{}" \;
    echo "ADD $folder $f" >> $dockerfile
    echo "ENV PYTHONPATH \$PYTHONPATH:$f" >> $dockerfile
done
echo "COMPILING TO BYTECODE AND DELETING SOURCE"
for f in $code; do
    echo "RUN python -OO -m compileall -f $f" >> $dockerfile
    echo "RUN find $f -name *.py ! -name __*__.py -delete" >> $dockerfile
done
# copy resources folders to buildfolder and create ADD commands
echo "COPYING RESOURCES"
while read f; do
    echo $f
    cp -r "$f" $buildfolder
    folder=`basename "$f"` # last folder in the path
    echo "ADD [ \"$folder\" , \"$f\" ]" >> $dockerfile
done <resource_folders.txt
# build image
echo "BUILDING IMAGE $EXEC_IMAGE"
docker build -t $EXEC_IMAGE --file $dockerfile $buildfolder

