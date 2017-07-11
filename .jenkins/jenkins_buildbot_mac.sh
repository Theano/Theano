#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10"
BASECOMPILEDIR=$HOME/.theano/buildbot_theano_mac
# Set test reports using nosetests xunit
XUNIT="--with-xunit --xunit-file="
SUITE="--xunit-testsuite-name="

export THEANO_FLAGS=init_gpu_device=cuda

# Set path for conda and cmake
export PATH="/Users/jenkins/miniconda2/bin:/usr/local/bin:$PATH"

# CUDA
export PATH=/Developer/NVIDIA/CUDA-8.0/bin${PATH:+:${PATH}}
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib\
                         ${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}

# Build libgpuarray
GPUARRAY_CONFIG="Release"
DEVICE=cuda
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

# Build the pygpu modules
(cd libgpuarray && python setup.py build_ext --inplace -I$LIBDIR/include -L$LIBDIR/lib)
ls $LIBDIR
mkdir $LIBDIR/lib/python
export PYTHONPATH=${PYTHONPATH}:$LIBDIR/lib/python
# Then install
(cd libgpuarray && python setup.py install --home=$LIBDIR)

python -c 'import pygpu; print(pygpu.__file__)'

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

set -x

# Fast run and float32
FILE=${BUILDBOT_DIR}/theano_python_fastrun_f32_tests.xml
NAME=mac_fastrun_f32
THEANO_FLAGS=$THEANO_FLAGS,base_compiledir=$BASECOMPILEDIR,mode=FAST_RUN,warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,floatX=float32,dnn.library_path=$HOME/cuda/lib,dnn.include_path=$HOME/cuda/include,gcc.cxxflags="-I/usr/local/cuda/include -I$LIBDIR/include -rpath $HOME/cuda/lib -L$HOME/.local/lib" python bin/theano-nose ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}