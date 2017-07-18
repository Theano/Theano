#!/bin/bash

# Script for Jenkins continuous integration testing on macOS

# Print commands as they are executed
set -x

# Copy cache from master
BASECOMPILEDIR=$HOME/.theano/pr_theano_mac
rsync -a $HOME/.theano/buildbot_theano_mac/ $BASECOMPILEDIR

# Set path for conda and cmake
export PATH="/Users/jenkins/miniconda2/bin:/usr/local/bin:$PATH"

# CUDA
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib\
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

# Testing theano
THEANO_PARAM="theano --with-timer --timer-top-n 10 --with-xunit --xunit-file=theano_mac_pr_tests.xml"
FLAGS=init_gpu_device=$DEVICE,gpuarray.preallocate=1000,mode=FAST_RUN,on_opt_error=raise,on_shape_error=raise,cmodule.age_thresh_use=604800,base_compiledir=$BASECOMPILEDIR,dnn.library_path=$HOME/cuda/lib,dnn.include_path=$HOME/cuda/include,gcc.cxxflags="-I/usr/local/cuda/include -I$LIBDIR/include -rpath $HOME/cuda/lib -L$HOME/.local/lib"
THEANO_FLAGS=${FLAGS} python bin/theano-nose ${THEANO_PARAM}
