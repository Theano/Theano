
#!/bin/bash

# Script for Jenkins continuous integration testing on macOS

# Print commands as they are executed
set -x

# Copy cache from master
BASECOMPILEDIR=$HOME/.theano/pr_theano_mac
# rsync -a --delete $HOME/.theano/buildbot_theano_mac/ $BASECOMPILEDIR

# Set path for conda and cmake
export PATH="/Users/jenkins/miniconda2/bin:/usr/local/bin:$PATH"

# CUDA
export PATH=/usr/local/cuda/bin:${PATH}
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:${DYLD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${HOME}/cuda/include:${CPLUS_INCLUDE_PATH}

# CUDNN
export CUDNNPATH=${HOME}/cuda

# Build libgpuarray
GPUARRAY_CONFIG="Release"
DEVICE=cuda
LIBDIR=${WORKSPACE}/local

# Make fresh clones of libgpuarray (with no history since we don't need it)
rm -rf libgpuarray
git clone -b `cat .jenkins/gpuarray-branch` "https://github.com/Theano/libgpuarray.git"

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

export DYLD_LIBRARY_PATH=${LIBDIR}/lib:${DYLD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=:${LIBDIR}/include:${CPLUS_INCLUDE_PATH}

python -c 'import pygpu; print(pygpu.__file__)'

# Testing theano
THEANO_PARAM="theano --with-timer --timer-top-n 10 --with-xunit --xunit-file=theano_mac_pr_tests.xml"
FLAGS=init_gpu_device=$DEVICE,gpuarray.preallocate=1000,mode=FAST_RUN,on_opt_error=raise,on_shape_error=raise,cmodule.age_thresh_use=604800,base_compiledir=$BASECOMPILEDIR,dnn.base_path=${CUDNNPATH},gcc.cxxflags="-L${LIBDIR}/lib"
THEANO_FLAGS=${FLAGS} python bin/theano-nose ${THEANO_PARAM}
