#!/bin/bash -e

THIS_DIR=$(cd $(dirname $0); pwd)
PREFIX=${PREFIX:-"/usr/local"}
MAKE=${MAKE:-"make"}
MAKEFLAGS=${MAKEFLAGS:-""}
SUDO=${SUDO:-""} 
CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}

echo "=== Installing requirements..."
cat requirement-rtd.txt | xargs -n1 ${SUDO} pip install --upgrade --no-cache-dir

echo "=== Building pycuda ..."
cd ${THIS_DIR}/third_party/pycuda \
    && ./configure.py --cuda-root=${CUDA_HOME} \
    && ${SUDO} ${MAKE} ${MAKEFLAGS} install \
    && ${SUDO} ldconfig

echo "=== Building skcuda ..."
cd ${THIS_DIR}/third_party/scikit-cuda \
    && ${SUDO} pip install .

echo "=== Building gpuarray ..."
cd ${THIS_DIR}/libgpuarray \
    && cmake -E make_directory build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    && VERBOSE=1 ${SUDO} ${MAKE} ${MAKEFLAGS} install \
    && cd .. \
    && ${SUDO} pip install . \
    && ${SUDO} ldconfig

echo "=== Installing Theano ..."
cd ${THIS_DIR} \
    && ${SUDO} pip install -e .

echo "=== Finished installing Theano."

