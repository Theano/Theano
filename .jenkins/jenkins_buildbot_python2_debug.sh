#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10 -v"
export MKL_THREADING_LAYER=GNU
export THEANO_FLAGS=init_gpu_device=cuda

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include/:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

GPUARRAY_CONFIG="Release"
DEVICE=cuda
LIBDIR=${WORKSPACE}/local

# Make fresh clones of libgpuarray (with no history since we don't need it)
rm -rf libgpuarray
git clone "https://github.com/Theano/libgpuarray.git"

# Clean up previous installs (to make sure no old files are left)
rm -rf $LIBDIR
mkdir $LIBDIR

# Build libgpuarray
mkdir libgpuarray/build
(cd libgpuarray/build && cmake .. -DCMAKE_BUILD_TYPE=${GPUARRAY_CONFIG} -DCMAKE_INSTALL_PREFIX=$LIBDIR && make)

# Finally install
(cd libgpuarray/build && make install)

# Export paths
export CPATH=$CPATH:$LIBDIR/include
export LIBRARY_PATH=$LIBRARY_PATH:$LIBDIR/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBDIR/lib

# Build the pygpu modules
(cd libgpuarray && python setup.py build_ext --inplace -I$LIBDIR/include -L$LIBDIR/lib)
ls $LIBDIR
mkdir $LIBDIR/lib/python
export PYTHONPATH=${PYTHONPATH}:$LIBDIR/lib/python
# Then install
(cd libgpuarray && python setup.py install --home=$LIBDIR)

python -c 'import pygpu; print(pygpu.__file__)'

# nosetests xunit for test profiling
XUNIT="--with-xunit --xunit-file="
SUITE="--xunit-testsuite-name="

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

BASE_COMPILEDIR=$HOME/.theano/buildbot_theano_python2_debug
ROOT_CWD=$WORKSPACE/nightly_build
FLAGS=base_compiledir=$BASE_COMPILEDIR
COMPILEDIR=`THEANO_FLAGS=$FLAGS python -c "from __future__ import print_function; import theano; print(theano.config.compiledir)"`
NOSETESTS=bin/theano-nose
export PYTHONPATH=${WORKSPACE}:$PYTHONPATH

echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l

# Exit if theano.gpuarray import fails
python -c "import theano.gpuarray; theano.gpuarray.use('${DEVICE}')" || { echo 'theano.gpuarray import failed, exiting'; exit 1; }

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

# Only use elements in the cache for < 7 days
FLAGS=${FLAGS},cmodule.age_thresh_use=604800

# Enable magma GPU library
FLAGS=${FLAGS},magma.enabled=true

#we change the seed and record it everyday to test different combination. We record it to be able to reproduce bug caused by different seed. We don't want multiple test in DEBUG_MODE each day as this take too long.
seed=$RANDOM
echo "Executing tests with mode=DEBUG_MODE with seed of the day $seed"
FILE=${ROOT_CWD}/theano_debug_tests.xml
echo "THEANO_FLAGS=${FLAGS},unittests.rseed=$seed,mode=DEBUG_MODE,DebugMode.check_strides=0,DebugMode.patience=3,DebugMode.check_preallocated_output= ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE}"
date
NAME=python2_debug
THEANO_FLAGS=${FLAGS},unittests.rseed=$seed,mode=DEBUG_MODE,DebugMode.check_strides=0,DebugMode.patience=3,DebugMode.check_preallocated_output= ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}

echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo
