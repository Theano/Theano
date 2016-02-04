from __future__ import absolute_import, print_function, division
import numpy
import theano
import theano.tensor as tt
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = tt.matrix("x")
y = tt.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print("Initial model:")
print(w.get_value(), b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + tt.exp(-tt.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                      # The prediction thresholded
xent = -y * tt.log(p_1) - (1 - y) * tt.log(1 - p_1)  # Cross-entropy loss
cost = xent.mean() + 0.01 * (w ** 2).sum()  # The cost to minimize
gw, gb = tt.grad(cost, [w, b])

# Compile
train = theano.function(
    inputs=[x, y],
    outputs=[prediction, xent],
    updates=[(w, w - 0.1 * gw),
             (b, b - 0.1 * gb)],
    name='train')

predict = theano.function(inputs=[x], outputs=prediction,
                          name='predict')

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value(), b.get_value())
print("target values for D:", D[1])
print("prediction on D:", predict(D[0]))
