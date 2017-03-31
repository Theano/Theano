#!/bin/bash

cd ../third_party/convnet-benchmarks/theano
exec python benchmark_imagenet.py --arch=overfeat
