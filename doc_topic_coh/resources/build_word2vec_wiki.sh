#!/usr/bin/env bash
program=/data/code/word2vec/trunk/word2vec
infile=/data/resources/doc_coherence/dump_wiki_en20150602_GtAR_tokenized.txt

numIter=7
cbow=$1
outfile="/datafast/word2vec/word2vec.wiki.cbow$cbow.en20150602.vectors.bin"
vocabfile="/datafast/word2vec/word2vec.wiki.cbow$cbow.en20150602.vocab.txt"

# old options, for skip-gram
#options="-train $infile -output $outfile -save-vocab $vocabfile -binary 1 -threads 3 -cbow 0 -hs 0 \
#         -size 300 -window $window -sample 1e-4 -negative 20 -iter 7"

# minimal modification of options used to create pre-trained google news embeddings
options="-train $infile -output $outfile -save-vocab $vocabfile -iter $numIter -binary 1 -threads 3 \
         -cbow $1 -hs 0 -size 300 -window 5 -negative 6 -sample 1e-5 -min-count 5"
#echo $options
$program $options
