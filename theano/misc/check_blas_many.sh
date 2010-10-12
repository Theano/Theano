#!/bin/bash

python misc/check_blas.py --print_only

cat /proc/cpuinfo |grep "model name" |uniq
cat /proc/cpuinfo |grep processor
free
uname -a

t0=`THEANO_FLAGS=blas.ldflags= OMP_NUM_THREADS=1 time python misc/check_blas.py --quiet`
t1=`OMP_NUM_THREADS=1 time python misc/check_blas.py --quiet`
t2=`OMP_NUM_THREADS=2 time python misc/check_blas.py --quiet`
t4=`OMP_NUM_THREADS=4 time python misc/check_blas.py --quiet`
t8=`OMP_NUM_THREADS=8 time python misc/check_blas.py --quiet`

echo "numpy gemm took: $t0"
echo "theano gemm 1 thread took: $t1"
echo "theano gemm 2 thread took: $t2"
echo "theano gemm 4 thread took: $t4"
echo "theano gemm 8 thread took: $t8"

#Fred to test distro numpy at LISA: PYTHONPATH=/u/bastienf/repos:/usr/lib64/python2.5/site-packages THEANO_FLAGS=blas.ldflags= OMP_NUM_THREADS=8 time python misc/check_blas.py