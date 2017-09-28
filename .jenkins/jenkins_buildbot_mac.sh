#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano -e gpuarray --with-timer --timer-top-n 10"
BASECOMPILEDIR=$HOME/.theano/buildbot_theano_mac
# Set test reports using nosetests xunit
XUNIT="--with-xunit --xunit-file="
SUITE="--xunit-testsuite-name="

# Set path for conda and cmake
export PATH="/Users/jenkins/miniconda2/bin:/usr/local/bin:$PATH"

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

set -x

# Fast run and float32
FILE=${BUILDBOT_DIR}/theano_python_fastrun_f32_tests.xml
NAME=mac_fastrun_f32
THEANO_FLAGS=$THEANO_FLAGS,base_compiledir=$BASECOMPILEDIR,mode=FAST_RUN,warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,floatX=float32 python bin/theano-nose ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
