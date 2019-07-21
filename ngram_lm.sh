#!/bin/sh
# $1=inputDir, $2=outputBaseDir, $3=numClasses $4=mixing_lambda


# Trigram Language Model with KN Smoothing
NGRAM_OUT_DIR=$2/ngram
mkdir -p $NGRAM_OUT_DIR

ngram-count -order 3 -wbdiscount1 -kndiscount2 -kndiscount3 -interpolate -text $1/train.txt -lm $NGRAM_OUT_DIR/3gram.lm
ngram -lm $NGRAM_OUT_DIR/3gram_kn_interp.lm -ppl $1/train.txt -debug 0 > $NGRAM_OUT_DIR/3gram.train.out
ngram -lm $NGRAM_OUT_DIR/3gram_kn_interp.lm -ppl $1/test.txt -debug 0 > $NGRAM_OUT_DIR/3gram.test.out
ngram -lm $NGRAM_OUT_DIR/3gram_kn_interp.lm -ppl $1/valid.txt -debug 0 > $NGRAM_OUT_DIR/3gram.valid.out
