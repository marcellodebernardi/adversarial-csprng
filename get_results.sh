#!/bin/bash
# this script runs the result file decoder, runs the dieharder testsuite on the
# decoded results, and sends the files to the specified location using scp
# CLI arguments are as follows:
# $1 folder within medb10/results/ into which text should be stored, e.g. predictive/01/

echo "--> Preparing to run dieharder ... "
# decode
cd ./src/utils/
python3 decode_dieharder.py ../../output/sequences/0_jerry.txt
python3 decode_dieharder.py ../../output/sequences/1_jerry.txt
python3 decode_dieharder.py ../../output/sequences/0_janice.txt
python3 decode_dieharder.py ../../output/sequences/1_janice.txt
# remove originals
cd ../../output/sequences/
rm 0_jerry.txt
rm 1_jerry.txt
rm 0_janice.txt
rm 1_janice.txt
# run dieharder
dieharder -g 202 -f 1_jerry_dieharder.txt -a -m 0.01 > result_jerry_1.txt
dieharder -g 202 -f 0_jerry_dieharder.txt -a -m 0.01 > result_jerry_0.txt
dieharder -g 202 -f 1_janice_dieharder.txt -a -m 0.01 > result_jerry_1.txt
dieharder -g 202 -f 0_janice_dieharder.txt -a -m 0.01 > result_jerry_1.txt
# scp files to frank
cd ../
scp -r ./* medb10@frank.eecs.qmul.ac.uk:/homes/medb10/results/$1
echo "[DONE]"