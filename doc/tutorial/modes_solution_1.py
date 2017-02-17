#!/usr/bin/env python
# Theano tutorial
# Solution to Exercise in section 'Configuration Settings and Compiling Modes'

from __future__ import absolute_import, print_function, division
import numpy
import theano
import theano.tensor as tt

theano.config.floatX = 'float32'

rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats).astype(theano.config.floatX),
rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
training_steps = 10000

# Declare Theano symbolic variables
x = tt.matrix("x")
y = tt.vector("y")
w = theano.shared(rng.randn(feats).astype(theano.config.floatX), name="w")
b = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), name="b")
x.tag.test_value = D[0]
y.tag.test_value = D[1]
#print "Initial model:"
#print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1 = 1 / (1 + tt.exp(-tt.dot(x, w) - b))  # Probability of having a one
prediction = p_1 > 0.5  # The prediction that is done: 0 or 1
xent = -y * tt.log(p_1) - (1 - y) * tt.log(1 - p_1)  # Cross-entropy
cost = tt.cast(xent.mean(), 'float32') + \
       0.01 * (w ** 2).sum()  # The cost to optimize
gw, gb = tt.grad(cost, [w, b])

# Compile expressions to functions
train = theano.function(
            inputs=[x, y],
            outputs=[prediction, xent],
            updates={w: w - 0.01 * gw, b: b - 0.01 * gb},
            name="train")
predict = theano.function(inputs=[x], outputs=prediction,
            name="predict")

if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
train.maker.fgraph.toposort()]):
    print('Used the cpu')
elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
train.maker.fgraph.toposort()]):
    print('Used the gpu')
else:
    print('ERROR, not able to tell if theano used the cpu or the gpu')
    print(train.maker.fgraph.toposort())

for i in range(training_steps):
    pred, err = train(D[0], D[1])
#print "Final model:"
#print w.get_value(), b.get_value()

print("target values for D")
print(D[1])

print("prediction on D")
print(predict(D[0]))
