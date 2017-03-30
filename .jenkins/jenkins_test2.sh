#!/bin/bash

# Script for Jenkins continuous integration testing of gpu backends

# Print commands as they are executed
set -x

# Anaconda python
export PATH=/usr/local/miniconda2/bin:$PATH

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

echo "===== Testing old theano.sandbox.cuda backend"

THEANO_CUDA_TESTS="theano/sandbox/cuda/tests \
            theano/misc/tests/test_pycuda_example.py \
            theano/misc/tests/test_pycuda_theano_simple.py \
            theano/misc/tests/test_pycuda_utils.py \
            theano/tensor/tests/test_opt.py:TestCompositeCodegen \
            theano/tensor/tests/test_opt.py:test_shapeoptimizer \
            theano/tensor/tests/test_opt.py:test_fusion \
            theano/compile/tests/test_debugmode.py:Test_preallocated_output \
            theano/sparse/tests/test_basic.py:DotTests \
            theano/sandbox/tests/test_multinomial.py:test_gpu_opt \
            theano/sandbox/tests/test_rng_mrg.py:test_consistency_GPU_serial \
            theano/sandbox/tests/test_rng_mrg.py:test_consistency_GPU_parallel \
            theano/sandbox/tests/test_rng_mrg.py:test_GPU_nstreams_limit \
            theano/sandbox/tests/test_rng_mrg.py:test_overflow_gpu_old_backend \
            theano/scan_module/tests/test_scan.py:T_Scan_Cuda"
THEANO_PARAM="${THEANO_CUDA_TESTS} --with-timer --timer-top-n 10 --with-xunit --xunit-file=theanocuda_tests.xml"
FLAGS="mode=FAST_RUN,init_gpu_device=gpu,floatX=float32"
THEANO_FLAGS=${FLAGS} bin/theano-nose ${THEANO_PARAM}

echo "===== Testing gpuarray backend"

GPUARRAY_CONFIG="Release"
DEVICE=cuda0
LIBDIR=${WORKSPACE}/local

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

# Testing theano (the gpuarray parts)
THEANO_GPUARRAY_TESTS="theano/gpuarray/tests \
                       theano/sandbox/tests/test_rng_mrg.py:test_consistency_GPUA_serial \
                       theano/sandbox/tests/test_rng_mrg.py:test_consistency_GPUA_parallel \
                       theano/sandbox/tests/test_rng_mrg.py:test_GPUA_full_fill \
                       theano/scan_module/tests/test_scan.py:T_Scan_Gpuarray \
                       theano/scan_module/tests/test_scan_checkpoints.py:TestScanCheckpoint.test_memory"
FLAGS="init_gpu_device=$DEVICE,gpuarray.preallocate=1000,mode=FAST_RUN,on_opt_error=raise,on_shape_error=raise,cmodule.age_thresh_use=604800"
THEANO_FLAGS=${FLAGS} time nosetests --with-xunit --xunit-file=theanogpuarray_tests.xml ${THEANO_GPUARRAY_TESTS}
