# intel software optimization for theano*
---

This repo is dedicated to improving Theano performance on CPU, especially in Intel® Xeon® and Intel® Xeon Phi™ processors.

**Key Features**
  * New backend of Intel® MKL (version >= 2017.0 which includes neural network primitives)
  * Further graph optimizations
  * CPU friendly OPs
  * Switch to Intel® MKL backend automatically in Intel®  Architecture
  * Out-of-box performance improvements and good portability

**Performance Tips**
  * Add bias after convolution to archieve high performance since this sub-graph can be replaced with MKL OP 
  * Use group convolution OP, [AbstractConvGroup](https://github.com/intel/Theano/blob/master/theano/sandbox/mkl/mkl_conv.py)
  * Use New MKL OP: [LRN](https://github.com/intel/Theano/blob/dev/theano/tensor/nnet/lrn.py)
  * Intel® Xeon Phi™ Environment Setting, example as below

        #!/bin/sh
        export KMP_BLOCKTIME=1
        export KMP_AFFINITY=verbose, granularity=core,noduplicates,compact,0,0
        export OMP_NUM_THREADS=68
        export MKL_DYNAMIC=false
        pytthon xxx.py

**Branch Information**
  * master, stable and fully tested version based on 0.9dev2 with Intel® MKL backend
  * nomkl-optimized, based on 0.9.0dev1 with generic optimizations
  * others, experimental codes for different applications which may be merged into master and/or deleted soon

**Installation**

  * Quick Commands

    ```
    git clone https://github.com/intel/theano.git intel-theano
    cd intel-theano
    python setup.py build
    python setup.py install --user
    cp intel-theano/theanorc_icc_mkl ~/.theanorc
    # run benchmark
    democase/alexnet/benchmark.sh
    ```

  * Install Guide (recommend to go througth this document and set up optimized softwares)
    https://github.com/intel/theano/blob/master/Install_Guide.pdf


**Other Optimized Software**
  * Self-contained MKL in [here](https://github.com/01org/mkl-dnn/releases)
  * Optimized Numpy in [here](https://github.com/pcs-theano/numpy)

---
>\* Other names and trademarks may be claimed as the property of others.
