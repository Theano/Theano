import theano.config as config
from theano.scalar import float32, float64
from theano.tensor import fscalar, fvector, fmatrix, frow, fcol, ftensor3, ftensor4, dscalar, dvector, dmatrix, drow, dcol, dtensor3, dtensor4

def set_floatX(dtype = config.floatX):
  """ add the xmatrix, xvector, xscalar etc. aliases to theano.tensor
  """
  print "set_floatX",dtype
  config.floatX = dtype
  if dtype == 'float32': prefix = 'f'
  elif dtype == 'float64' : prefix = 'd'
  else: raise Exception("Bad param in set_floatX(%s). Only float32 and float64 are supported"%config.floatX)

  #tensor.scalar stuff
  globals()['floatX'] = globals()[dtype]
#  convert_to_floatX = Cast(floatX, name='convert_to_floatX')


  #tensor.tensor stuff
  for symbol in ('scalar', 'vector', 'matrix', 'row', 'col','tensor3','tensor4'):
     globals()['x'+symbol] = globals()[prefix+symbol]
  #_convert_to_floatX = _conversion(elemwise.Elemwise(scal.convert_to_floatX), 'floatX')
  




