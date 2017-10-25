#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10"

export MKL_THREADING_LAYER=GNU
# Set test reports using nosetests xunit
XUNIT="--with-xunit --xunit-file="
SUITE="--xunit-testsuite-name="

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

FILE=${BUILDBOT_DIR}/theano_python32bit_tests.xml
set -x
NAME=python2_32bit
THEANO_FLAGS=device=cpu,force_device=true,lib.amdlibm=False,compiledir=$HOME/.theano/buildbot_theano_python2_32bit python bin/theano-nose ${THEANO_PARAM} ${XUNIT}${FILE}  ${SUITE}${NAME}
