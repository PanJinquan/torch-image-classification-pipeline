#!/usr/bin/env bash

rm -rf codes
mkdir codes
cp -r ../configs ../core ../data  ../evaluation ../utils ../work_space ../test.py  ../train.py ../train_dali.py ../train_distributed.sh ../requirements.txt ../README.md codes
find . -name "__pycache__" | xargs rm -rf
find . -name "*.py[cod]" | xargs rm -rf

sudo docker build . -t face-recognize-pytorch:v2
