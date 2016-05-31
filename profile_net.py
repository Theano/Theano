import sys
import math
import numpy
import theano
import theano.tensor as tensor
from theano.tensor.signal import downsample

def relu(x, x_shape):
    return tensor.nnet.relu(x), x_shape, []

def log_softmax(x, x_shape):
    xx = x[:,:,0,0]
    m = tensor.max(xx, axis=1)
    s = tensor.sum(tensor.exp(xx - m[:,None]), axis=1)
    d = m + tensor.log(s)
    return xx - d[:,None], (x_shape[0], x_shape[1]), []
    
def conv(x, x_shape, shape, stride=(1,1)):
    omega_std = math.sqrt(2.0 / (shape[2]*shape[3]*shape[0]))
    omega = theano.shared(numpy.asarray(numpy.random.normal(0.0, omega_std, size=shape), dtype=theano.config.floatX), borrow=True)
    y = tensor.nnet.conv2d(input=x, filters=omega, filter_shape=shape, subsample=stride, input_shape=x_shape, border_mode="half")
    y_shape = (x_shape[0], shape[0], x_shape[2] // stride[0], x_shape[3] // stride[1])
    print("conv:", x_shape, "->", y_shape)
    return y, y_shape, [omega]
    
def batch_norm(x, x_shape, momentum=0.9, eps=1e-4):
    x_mean = x.mean(axis=[0,2,3])
    x_std = x.std(axis=[0,2,3]) + eps
    gamma = theano.shared(numpy.asarray(numpy.ones((x_shape[1],)), dtype=theano.config.floatX), borrow=True)
    bias = theano.shared(numpy.asarray(numpy.zeros((x_shape[1],)), dtype=theano.config.floatX), borrow=True)
    y = tensor.nnet.bn.batch_normalization(x, gamma[None,:,None,None], bias[None,:,None,None], x_mean[None,:,None,None], x_std[None,:,None,None], mode = "low_mem")
    return y, x_shape, [gamma, bias] 
        
#def batch_norm(x, x_shape, momentum=0.9, eps=1e-4):
#    return x, x_shape, []
        
def resnet_bottleneck(x, x_shape, features, stride=1):

    b = features // 4
    y0, y0_shape, p0 = batch_norm(x, x_shape)
    y1, y1_shape, p1 = relu(y0, y0_shape)
    y2, y2_shape, p2 = conv(y1, y1_shape, (b, x_shape[1], 1, 1), (stride, stride))
    y3, y3_shape, p3 = batch_norm(y2, y2_shape)
    y4, y4_shape, p4 = relu(y3, y3_shape)
    y5, y5_shape, p5 = conv(y4, y4_shape, (b, b, 3, 3), (1,1))
    y6, y6_shape, p6 = batch_norm(y5, y5_shape)
    y7, y7_shape, p7 = relu(y6, y6_shape)
    y8, y8_shape, p8 = conv(y7, y7_shape, (features, b, 1, 1), (1,1))

    params = p0+p1+p2+p3+p4+p5+p6+p7+p8
    if x_shape != y8_shape:
        y9, y9_shape, p9 = conv(y1, y1_shape, (features, x_shape[1], 1, 1), (stride, stride))
        params += p9
        y = y8 + y9
    else:
        y = y8 + x

    print("resnet:", x_shape, "->", y8_shape)
    return y, y8_shape, params
        
def build_resnet(x, x_shape, features = 64):

    y0, y0_shape, p0 = conv(x, x_shape, (features,3,3,3))
    y1, y1_shape, p1 = batch_norm(y0, y0_shape)
    y2, y2_shape, _ = relu(y1, y1_shape)

    #first resnet block
    y3, y3_shape, p3 = resnet_bottleneck(y2, y2_shape, features)
    y4, y4_shape, p4 = resnet_bottleneck(y3, y3_shape, features)
    y5, y5_shape, p5 = resnet_bottleneck(y4, y4_shape, features)

    #second resnet block
    y6, y6_shape, p6 = resnet_bottleneck(y5, y5_shape, 2*features, 2)
    y7, y7_shape, p7 = resnet_bottleneck(y6, y6_shape, features)
    y8, y8_shape, p8 = resnet_bottleneck(y7, y7_shape, features)

    #third resnet block
    y9, y9_shape, p9 = resnet_bottleneck(y8, y8_shape, 4*features, 2)
    y10, y10_shape, p10 = resnet_bottleneck(y9, y9_shape, features)
    y11, y11_shape, p11 = resnet_bottleneck(y10, y10_shape, features)

    #8x8 average pooling 
    y12 = downsample.max_pool_2d(y11, (8,8), mode="average_inc_pad", ignore_border=True)
    y12_shape=(y11_shape[0], y11_shape[1],1,1)
    
    #regression layer (log-softmax)
    y13, y13_shape, p13 = conv(y12, y12_shape, (10, y12_shape[1], 1, 1))
    y14, y14_shape, _ = log_softmax(y13, y13_shape)
        
    return y14, y14_shape, p0+p1+p3+p4+p5+p6+p7+p8+p9+p10+p11+p13
    
    
def main():

    print("testing resnet with 34 convolution layers")
    
    #hyper parameters
    x = tensor.tensor4()
    x_shape = (32, 3, 32, 32)
    y = tensor.matrix()
    y_shape = (32, 10)
    lr = tensor.scalar()
    decay=0.0001
    momentum=0.9
    
    print("building function graph")
    yy, yy_shape, params = build_resnet(x, x_shape)
    cost = tensor.mean(tensor.sum(yy*y, axis=1))

    #SGD with momentum
    updates=[]
    for p in params:
        m = theano.shared(numpy.asarray(p.get_value(borrow=True)*0.0, dtype=theano.config.floatX), broadcastable=p.broadcastable, borrow=True)
        gradient = tensor.grad(cost, p) + decay*p
        m_update = momentum*m + (1.0 - momentum)*gradient
        p_update = p - lr*m_update
        updates += [(p, p_update), (m, m_update)]
        
    print("compile graph")
    f = theano.function([x,y,lr], cost, updates=updates)
    
    #test random data:
    print("test")
    xx = numpy.random.uniform(0, 1.0, x_shape).astype(theano.config.floatX)
    yy = numpy.random.uniform(0, 1.0, (x_shape[0], 10)).astype(theano.config.floatX)
    c = f(xx, yy, 0.1)

    print("done")
    
if __name__ == '__main__':
    main()