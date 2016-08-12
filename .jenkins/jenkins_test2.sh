#!/bin/bash

# Script for Jenkins continuous integration testing of gpu backends
# Get environment from worker, necessary for CUDA
source ~/.bashrc

echo "===== Testing old theano.sandbox.cuda backend"

PARTS="theano/sandbox/cuda"
THEANO_PARAM="${PARTS} --with-timer --timer-top-n 10"
FLAGS="mode=FAST_RUN,init_gpu_device=gpu,floatX=float32"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}

echo "===== Testing gpuarray backend"

GPUARRAY_CONFIG="Release"
DEVICE=cuda0
LIBDIR=~/tmp/local

# Make fresh clones of libgpuarray (with no history since we don't need it)
rm -rf libgpuarray
git clone --depth 1 "https://github.com/Theano/libgpuarray.git"

# Clean up previous installs (to make sure no old files are left) 
rm -rf $LIBDIR
mkdir $LIBDIR

# Build libgpuarray
mkdir libgpuarray/build
(cd libgpuarray/build && cmake .. -DCMAKE_BUILD_TYPE=${GPUARRAY_CONFIG} -DCMAKE_INSTALL_PREFIX=$LIBDIR && make)

# Finally install                                                               
(cd libgpuarray/build && make install)

# Export paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBDIR/lib64/
export LIBRARY_PATH=$LIBRARY_PATH:$LIBDIR/lib64/
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

# Testing theano (the gpuarray parts)                                           
THEANO_GPUARRAY_TESTS="theano/gpuarray/tests theano/sandbox/tests/test_rng_mrg.py:test_consistency_GPUA_serial theano/sandbox/tests/test_rng_mrg.py:test_consistency_GPUA_parallel theano/scan_module/tests/test_scan.py:T_Scan_Gpuarray"
FLAGS="init_gpu_device=$DEVICE,gpuarray.preallocate=1000,mode=FAST_RUN"
THEANO_FLAGS=${FLAGS} time nosetests -v ${THEANO_GPUARRAY_TESTS}
