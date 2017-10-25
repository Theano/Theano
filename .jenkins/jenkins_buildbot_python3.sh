#!/bin/bash

BUILDBOT_DIR=$WORKSPACE/nightly_build
THEANO_PARAM="theano --with-timer --timer-top-n 10"
COMPILEDIR=$HOME/.theano/buildbot_theano_python3
export MKL_THREADING_LAYER=GNU

# Set test reports using nosetests xunit
XUNIT="--with-xunit --xunit-file="
SUITE="--xunit-testsuite-name="

export THEANO_FLAGS=init_gpu_device=cuda

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/usr/local/cuda/include/:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

# Build libgpuarray
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
(cd libgpuarray && python3 setup.py build_ext --inplace -I$LIBDIR/include -L$LIBDIR/lib)
ls $LIBDIR
mkdir $LIBDIR/lib/python
export PYTHONPATH=${PYTHONPATH}:$LIBDIR/lib/python
# Then install
(cd libgpuarray && python3 setup.py install --home=$LIBDIR)

python3 -c 'import pygpu; print(pygpu.__file__)'

mkdir -p ${BUILDBOT_DIR}
ls -l ${BUILDBOT_DIR}
echo "Directory of stdout/stderr ${BUILDBOT_DIR}"
echo
echo

# Exit if theano.gpuarray import fails
python -c "import theano.gpuarray; theano.gpuarray.use('${DEVICE}')" || { echo 'theano.gpuarray import failed, exiting'; exit 1; }

set -x

# Fast compile and float64
FILE=${BUILDBOT_DIR}/theano_python3_fastcompile_f64_tests.xml
NAME=python3_fastcompile_f64
THEANO_FLAGS=$THEANO_FLAGS,compiledir=$COMPILEDIR,mode=FAST_COMPILE,warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,magma.enabled=true,floatX=float64 python3 bin/theano-nose ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}

# Fast run and float32
FILE=${BUILDBOT_DIR}/theano_python3_fastrun_f32_tests.xml
NAME=python3_fastrun_f32
THEANO_FLAGS=$THEANO_FLAGS,compiledir=$COMPILEDIR,mode=FAST_RUN,warn.ignore_bug_before=all,on_opt_error=raise,on_shape_error=raise,magma.enabled=true,floatX=float32 python3 bin/theano-nose ${THEANO_PARAM} ${XUNIT}${FILE} ${SUITE}${NAME}
