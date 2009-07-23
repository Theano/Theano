

type_support.so : type_support.cu $(HOME)/cvs/lgcm/cuda_ndarray/cuda_ndarray.so
	nvcc -g -shared -I$(HOME)/cvs/lgcm/cuda_ndarray -I$(CUDA_ROOT)/include -I/usr/include/python2.6 -o type_support.so -Xcompiler -fPIC type_support.cu -L$(CUDA_ROOT)/lib -L$(HOME)/cvs/lgcm/cuda_ndarray -lcuda_ndarray

clean : 
	rm type_support.so
