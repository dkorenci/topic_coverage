#!/usr/bin/env bash
# script for building glove vectors on HrWac corpus
# based on the demo.sh script from Glove 1.2
# hyperparams are set as those given or best performing in the original article
#   with the exception of X_max which is set to 50, to acknowledge that HrWac
#   is approx. 5 times smaller than Gigaword+Wikipedia used in the original article

# params
CORPUS=$1
BUILDDIR=$2 # dir with glove executables
NUM_THREADS=$3
MEMORY=$4
PYEVAL=eval/python/evaluate.py

# input/output
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
SAVE_FILE=vectors
VERBOSE=2
BINARY=2

# hyperparams
VOCAB_MIN_COUNT=5
VOCAB_MAX_SIZE=200000
VECTOR_SIZE=300
MAX_ITER=100
X_MAX=50
# this values have best performances in the original article:
WINDOW_SIZE=10
SYMMETRIC=0

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -max-vocab $VOCAB_MAX_SIZE  -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]
  then
  $BUILDDIR/cooccur -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE -symmetric $SYMMETRIC < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
  then
    $BUILDDIR/shuffle -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
       if [[ $? -eq 0 && $PYEVAL ]]
       then
            python $PYEVAL
       fi
    fi
  fi
fi
