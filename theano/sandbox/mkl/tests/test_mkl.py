import theano
import theano.sandbox.mkl as mkl

print ('mkl_available: ' + str(mkl.mkl_available()))
print ('mkl_version: ' + str(mkl.mkl_version()))
