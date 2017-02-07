#!/bin/bash
set -e

(cd ../third_party/lasagne && pip install --no-deps -e .)

cd ../third_party/convnet-benchmarks
python benchmark_imagenet.py --arch=vgg
