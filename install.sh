#!/usr/bin/env bash

THIS_DIR=$(cd $(dirname $0); pwd)
PREFIX=${PREFIX:-"/usr/local"}
MAKE=${MAKE:-"make"}
SUDO=${SUDO:-""} 
CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}

echo "=== Installing requirements..."
cat requirement-rtd.txt | xargs -n1 ${SUDO} pip install --upgrade --no-cache-dir || exit 1

echo "=== Building pycuda ..."
cd ${THIS_DIR}/pycuda \
    && ./configure.py --cuda-root=${CUDA_HOME} \
    && VERBOSE=1 ${MAKE} \
    && ${SUDO} ${MAKE} install && ${SUDO} ldconfig || exit 1

echo "=== Building skcuda ..."
cd ${THIS_DIR}/skcuda \
    && ${SUDO} python setup.py install && ${SUDO} ldconfig || exit 1

echo "=== Building gpuarray ..."
cd ${THIS_DIR}/libgpuarray \
    && cmake -E make_directory build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    && VERBOSE=1 ${MAKE} \
    && VERBOSE=1 ${SUDO} ${MAKE} install && ${SUDO} ldconfig || exit 1

echo "=== Installing pygpu ..."
cd ${THIS_DIR}/libgpuarray \
    && python setup.py build_ext -L ${PREFIX}/lib -I ${PREFIX}/include \
    && ${SUDO} python setup.py install && ${SUDO} ldconfig || exit 1

cd ${THIS_DIR}

echo "=== Running setup.py ..."
python setup.py build_ext -L ${PREFIX}/lib -I ${PREFIX}/include \
    && ${SUDO} python setup.py install && ${SUDO} ldconfig || exit 1

# Install Theano via PIP
echo "=== Running setup.py ..."
${SUDO} pip install -e ${THIS_DIR} && ${SUDO} ldconfig || exit 1

echo "=== Finished installing Theano."

