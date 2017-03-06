#!/usr/bin/env bash

THIS_DIR=$(cd $(dirname $0); pwd)
# PREFIX=${PREFIX:-"${THIS_DIR}/install"}
PREFIX=${PREFIX:-"/usr/local"}
MAKE=${MAKE:-"make"}
SUDO=${SUDO:-""} 
CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}

echo "=== Installing requirements..."
cat requirement-rtd.txt | xargs -n1 ${SUDO} pip install --no-cache-dir || exit 1

echo "=== Builing pycuda ..."
cd ${THIS_DIR}/pycuda \
    && ./configure.py --cuda-root=${CUDA_HOME} --no-use-shipped-boost \
    && VERBOSE=1 make \
    && ${SUDO} make install || exit 1

echo "=== Builing gpuarray ..."
cd ${THIS_DIR}/libgpuarray \
    && cmake -E make_directory build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    && VERBOSE=1 ${MAKE} \
    && VERBOSE=1 ${SUDO} ${MAKE} install || exit 1

echo "=== Installing pygpu ..."
cd ${THIS_DIR}/libgpuarray \
    python setup.py build_ext \
    -L ${PREFIX}/lib -I ${PREFIX}/include \
    && ${SUDO} python setup.py install || exit 1


cd ${THIS_DIR}

echo "=== Running setup.py ..."
python setup.py build_ext \
    -L ${PREFIX}/lib -I ${PREFIX}/include \
    && ${SUDO} python setup.py install || exit 1

# Install Theano via PIP
echo "=== Running setup.py ..."
${SUDO} pip install -e ${THIS_DIR} || exit 1

echo "=== Finished installing Theano."

