#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10"
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

# name test suites
SUITE="--xunit-testsuite-name="

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

BASE_COMPILEDIR=$HOME/.theano/buildbot_theano_python2
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

# Enable magma GPU library
FLAGS=${FLAGS},magma.enabled=true

# Only use elements in the cache for < 7 days
FLAGS=${FLAGS},cmodule.age_thresh_use=604800

echo "Executing tests with mode=FAST_RUN"
NAME=python2_fastrun
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=cmodule.warn_no_version=True,${FLAGS},mode=FAST_RUN ${NOSETESTS} ${PROFILING} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=cmodule.warn_no_version=True,${FLAGS},mode=FAST_RUN ${NOSETESTS} ${PROFILING} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

echo "Executing tests with mode=FAST_RUN,floatX=float32"
NAME=python2_fastrun_float32
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=${FLAGS},mode=FAST_RUN,floatX=float32 ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=${FLAGS},mode=FAST_RUN,floatX=float32 ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo

echo "Executing tests with linker=vm,vm.lazy=True,floatX=float32"
NAME=python2_fastrun_float32_lazyvm
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
# force_device=True as it would be useless to test the gpuarray back-end.
echo "Executing tests with mode=FAST_COMPILE"
NAME=python2_fastcompile
FILE=${ROOT_CWD}/theano_${NAME}_tests.xml
echo "THEANO_FLAGS=${FLAGS},mode=FAST_COMPILE,force_device=True ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}"
date
THEANO_FLAGS=${FLAGS},mode=FAST_COMPILE,force_device=True ${NOSETESTS} ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}

echo "Number of elements in the compiledir:"
ls ${COMPILEDIR}|wc -l
echo
