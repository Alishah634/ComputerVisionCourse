#!/bin/bash
img0=./ece661_sample_images/1.jpg
img1=./ece661_sample_images/2.jpg
outDir=./ece661_sample_images

python3 superglue_ece661.py $img0 $img1 $outDir
