CUDA_NDARRAY?=$(HOME)/cvs/lgcm/cuda_ndarray
NUMPY_INCLUDE?=/usr/include/python2.6

type_support.so : type_support.cu $(CUDA_NDARRAY)/cuda_ndarray.so
	nvcc -g -shared -I$(CUDA_NDARRAY) -I$(CUDA_ROOT)/include -I$(NUMPY_INCLUDE) -o type_support.so -Xcompiler -fPIC type_support.cu -L$(CUDA_ROOT)/lib -L$(CUDA_NDARRAY) -lcuda_ndarray

clean : 
	rm type_support.so
