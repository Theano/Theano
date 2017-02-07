#!/bin/bash
set -e

(cd ../third_party/pylearn2 && pip install --no-deps -e .)
(cd ../third_party/scikits.cuda && pip install --no-deps -e .)

cd ../third_party/convnet-benchmarks
SKIP=legacy python pylearn2_benchmark.py
