#!/bin/bash

# Builds Theano master cache for Jenkins PR testing

BASECOMPILEDIR=$HOME/.theano

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

set -x

# Test theano core
PARTS="theano -e gpuarray"
THEANO_PARAM="${PARTS} --with-timer --timer-top-n 10 --with-xunit --xunit-file=theanocore_tests.xml"
FLAGS="mode=FAST_RUN,floatX=float32,on_opt_error=raise,on_shape_error=raise,base_compiledir=$BASECOMPILEDIR"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}

# Testing theano (the gpuarray parts)
THEANO_GPUARRAY_TESTS="theano/gpuarray/tests \
                       theano/scan_module/tests/test_scan.py:T_Scan_Gpuarray \
                       theano/scan_module/tests/test_scan_checkpoints.py:TestScanCheckpoint.test_memory"
FLAGS="init_gpu_device=$DEVICE,gpuarray.preallocate=1000,mode=FAST_RUN,on_opt_error=raise,on_shape_error=raise,base_compiledir=$BASECOMPILEDIR"
FLAGS=${FLAGS},magma.enabled=true # Enable magma GPU library
THEANO_FLAGS=${FLAGS} time nosetests --with-xunit --xunit-file=theanogpuarray_tests.xml ${THEANO_GPUARRAY_TESTS}

# copy master cache
rsync -a --delete $BASECOMPILEDIR/ $HOME/cache/master
