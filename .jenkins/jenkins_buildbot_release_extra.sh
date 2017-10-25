#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10"
export MKL_THREADING_LAYER=GNU
export THEANO_FLAGS=init_gpu_device=cuda

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

BASE_COMPILEDIR=$HOME/.theano/buildbot_theano_release
ROOT_CWD=$WORKSPACE/nightly_build
FLAGS=base_compiledir=$BASE_COMPILEDIR
COMPILEDIR=`THEANO_FLAGS=$FLAGS python -c "from __future__ import print_function; import theano; print(theano.config.compiledir)"`
NOSETESTS=bin/theano-nose
export PYTHONPATH=${WORKSPACE}:$PYTHONPATH

echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l

# We don't want warnings in the buildbot for errors already fixed.
FLAGS=${THEANO_FLAGS},warn.ignore_bug_before=all,$FLAGS

# We want to see correctly optimization/shape errors, so make make them raise an
# error.
FLAGS=on_opt_error=raise,$FLAGS
FLAGS=on_shape_error=raise,$FLAGS

# Ignore user device and floatX config, because:
#   1. Tests are intended to be run with device=cpu.
#   2. We explicitly add 'floatX=float32' in one run of the test suite below,
#      while we want all other runs to run with 'floatX=float64'.
FLAGS=${FLAGS},device=cpu,floatX=float64

echo "Executing tests with compute_test_value=ignore"
NAME=compute_test_value_ignore
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=${FLAGS},compute_test_value=ignore ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=${FLAGS},compute_test_value=ignore ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

echo "Executing tests with linker=vm,floatX=float32"
NAME=linker_vm_float32
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=${FLAGS},linker=vm,floatX=float32 ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=${FLAGS},linker=vm,floatX=float32 ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

echo "Executing tests with cxx="
NAME=cxx_none
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=${FLAGS},cxx= ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=${FLAGS},cxx= ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

echo "Executing tests with mode=FAST_RUN, no scipy"
NAME=python2_fastrun_noscipy
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=cmodule.warn_no_version=True,${FLAGS},mode=FAST_RUN ${NOSETESTS} ${PROFILING} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
source activate no_scipy
THEANO_FLAGS=cmodule.warn_no_version=True,${FLAGS},mode=FAST_RUN ${NOSETESTS} ${PROFILING} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
source deactivate
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

# Test shortcut to default test suite
echo "Running tests using theano.test()"
mkdir -p test_default
rm -rf test_default/*
cd test_default
NAME=import
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
EXTRA_ARGS='["--with-xunit", "--xunit-file='${FILE}$'", "'${SUITE}${NAME}'"]'
THEANO_FLAGS=base_compiledir=$BASE_COMPILEDIR python -c "import theano; theano.test(extra_argv=${EXTRA_ARGS})"
