#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10"
# Set test reports using nosetests xunit
XUNIT="--with-xunit --xunit-file="
export THEANO_FLAGS=init_gpu_device=gpu

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

FILE=${BUILDBOT_DIR}/theano_python3_tests.xml
set -x
PYTHONPATH= THEANO_FLAGS=$THEANO_FLAGS,compiledir=$WORKSPACE/compile/theano_compile_dir_theano_python3,mode=FAST_COMPILE,warn.ignore_bug_before=0.5,on_opt_error=raise,on_shape_error=raise python3 bin/theano-nose ${THEANO_PARAM} ${XUNIT}${FILE}
