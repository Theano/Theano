#!/bin/bash

# Anaconda python
export PATH=/usr/local/miniconda2/bin:$PATH

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10"

# nosetests xunit for test profiling
XUNIT="--with-xunit --xunit-file="

# name test suites
SUITE="--xunit-testsuite-name="

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

BASE_COMPILEDIR=$HOME/.theano/buildbot_theano_nogpp
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

echo "Executing tests with mode=FAST_RUN"
NAME=fastrun
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=cmodule.warn_no_version=True,${FLAGS},mode=FAST_RUN ${NOSETESTS} ${PROFILING} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=cmodule.warn_no_version=True,${FLAGS},mode=FAST_RUN ${NOSETESTS} ${PROFILING} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo
