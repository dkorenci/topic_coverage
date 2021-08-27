#!/usr/bin/env bash
program=./word2vec
infile=uspol_corpus.txt

numIter=20

cbow=$1
size=$2
window=$3

sig="cbow$cbow.size$size.window$window"
outfile="uspol_word2vec[$sig].bin"
vocabfile="uspol_word2vec_vocab[$sig].txt"

# minimal modification of options used to create pre-trained google news embeddings
options="-train $infile -output $outfile -save-vocab $vocabfile -iter $numIter -binary 1 -threads 3 \
         -cbow $cbow -hs 0 -size $size -window $window -negative 20 -sample 3e-3 -min-count 1"
#echo $options
$program $options
