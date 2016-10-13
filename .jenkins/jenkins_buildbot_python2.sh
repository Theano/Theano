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

# name test suites
SUITE="--xunit-testsuite-name="

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

echo "Executing tests with mode=FAST_RUN"
NAME=fastrun
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=cmodule.warn_no_version=True,${FLAGS},mode=FAST_RUN ${NOSETESTS} ${PROFILING} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=cmodule.warn_no_version=True,${FLAGS},mode=FAST_RUN ${NOSETESTS} ${PROFILING} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

echo "Executing tests with mode=FAST_RUN,floatX=float32"
NAME=fastrun_float32
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=${FLAGS},mode=FAST_RUN,floatX=float32 ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=${FLAGS},mode=FAST_RUN,floatX=float32 ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

echo "Executing tests with linker=vm,vm.lazy=True,floatX=float32"
NAME=fastrun_float32_lazyvm
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=${FLAGS},linker=vm,vm.lazy=True,floatX=float32 ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=${FLAGS},linker=vm,vm.lazy=True,floatX=float32 ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

#We put this at the end as it have a tendency to loop infinitly.
#Until we fix the root of the problem we let the rest run, then we can kill this one in the morning.
# with --batch=1000" # The buildbot freeze sometimes when collecting the tests to run
echo "Executing tests with mode=FAST_COMPILE"
NAME=fastcompile
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=${FLAGS},mode=FAST_COMPILE ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=${FLAGS},mode=FAST_COMPILE ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}

echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo
