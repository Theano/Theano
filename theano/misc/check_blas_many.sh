#!/bin/bash

python misc/check_blas.py --print_only

cat /proc/cpuinfo |grep "model name" |uniq
cat /proc/cpuinfo |grep processor
free
uname -a

TIME_PREFIX=time
VAR=OMP_NUM_THREADS
echo "numpy gemm take="
THEANO_FLAGS=blas.ldflags= $TIME_PREFIX python misc/check_blas.py --quiet
for i in 1 2 4 8
do
  export $VAR=$i
  x=`$TIME_PREFIX python misc/check_blas.py --quiet`
  echo "theano gemm with $VAR=$i took: ${x}s"
done

#Fred to test distro numpy at LISA: PYTHONPATH=/u/bastienf/repos:/usr/lib64/python2.5/site-packages THEANO_FLAGS=blas.ldflags= OMP_NUM_THREADS=8 time python misc/check_blas.py
