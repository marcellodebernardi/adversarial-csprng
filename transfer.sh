#!/bin/bash

# scp to frank
cd ./output/
scp -r ./* medb10@frank.eecs.qmul.ac.uk:/homes/medb10/results/$1