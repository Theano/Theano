#!/usr/bin/env bash

THIS_DIR=$(cd $(dirname $0); pwd)
SUDO=${SUDO:-""} 

cd ${THIS_DIR}
${SUDO} rm -fr build install libgpuarray/build opycuda/build 
${SUDO} git clean -f -X -d
git submodule foreach ${SUDO} git clean -f -X -d

