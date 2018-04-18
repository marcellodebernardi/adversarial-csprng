#!/bin/bash
# this script runs the result file decoder, runs the dieharder testsuite on the
# decoded results, and sends the files to the specified location using scp
# CLI arguments are as follows:
# $1 folder within medb10/results/ into which text should be stored, e.g. predictive/01/

echo "--> Decoding ... "

# decode
cd ./src/utils/
python3 decode_nist.py ../../output/sequences/0_jerry.txt
rm 0_jerry.txt
python3 decode_nist.py ../../output/sequences/1_jerry.txt
rm 1_jerry.txt
python3 decode_nist.py ../../output/sequences/0_janice.txt
rm 0_janice.txt
python3 decode_nist.py ../../output/sequences/1_janice.txt
rm 1_janice.txt

echo "[DONE]"