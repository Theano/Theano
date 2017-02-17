#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10"
export THEANO_FLAGS=init_gpu_device=gpu

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

# nosetests xunit for test profiling
XUNIT="--with-xunit --xunit-file="

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

BASE_COMPILEDIR=$WORKSPACE/compile/theano_compile_dir_theano_python2
ROOT_CWD=$WORKSPACE/nightly_build
FLAGS=base_compiledir=$BASE_COMPILEDIR
COMPILEDIR=`THEANO_FLAGS=$FLAGS python -c "from __future__ import print_function; import theano; print(theano.config.compiledir)"`
NOSETESTS=bin/theano-nose
export PYTHONPATH=${WORKSPACE}:$PYTHONPATH

echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l

# We don't want warnings in the buildbot for errors already fixed.
FLAGS=${THEANO_FLAGS},warn.argmax_pushdown_bug=False,warn.gpusum_01_011_0111_bug=False,warn.sum_sum_bug=False,warn.sum_div_dimshuffle_bug=False,warn.subtensor_merge_bug=False,$FLAGS

# We want to see correctly optimization/shape errors, so make make them raise an
# error.
FLAGS=on_opt_error=raise,$FLAGS
FLAGS=on_shape_error=raise,$FLAGS

# Ignore user device and floatX config, because:
#   1. Tests are intended to be run with device=cpu.
#   2. We explicitly add 'floatX=float32' in one run of the test suite below,
#      while we want all other runs to run with 'floatX=float64'.
FLAGS=${FLAGS},device=cpu,floatX=float64

#we change the seed and record it everyday to test different combination. We record it to be able to reproduce bug caused by different seed. We don't want multiple test in DEBUG_MODE each day as this take too long.
seed=$RANDOM
echo "Executing tests with mode=DEBUG_MODE with seed of the day $seed"
FILE=${ROOT_CWD}/theano_debug_tests.xml
echo "THEANO_FLAGS=${FLAGS},unittests.rseed=$seed,mode=DEBUG_MODE,DebugMode.check_strides=0,DebugMode.patience=3,DebugMode.check_preallocated_output= ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE}"
date
THEANO_FLAGS=${FLAGS},unittests.rseed=$seed,mode=DEBUG_MODE,DebugMode.check_strides=0,DebugMode.patience=3,DebugMode.check_preallocated_output= ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE}

echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo
