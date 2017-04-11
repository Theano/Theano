#!/usr/bin/env python
# Theano tutorial
# Solution to Exercise in section 'Using the GPU'


# 1. Raw results


from __future__ import absolute_import, print_function, division
import numpy as np
import theano
import theano.tensor as tt

theano.config.floatX = 'float32'

rng = np.random

N = 400
feats = 784
D = (rng.randn(N, feats).astype(theano.config.floatX),
    rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
training_steps = 10000

# Declare Theano symbolic variables
x = theano.shared(D[0], name="x")
y = theano.shared(D[1], name="y")
w = theano.shared(rng.randn(feats).astype(theano.config.floatX), name="w")
b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name="b")
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
            inputs=[],
            outputs=[prediction, xent],
            updates=[(w, w - 0.01 * gw), (b, b - 0.01 * gb)],
            name="train")
predict = theano.function(inputs=[], outputs=prediction,
            name="predict")

if any([n.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for n in
train.maker.fgraph.toposort()]):
    print('Used the cpu')
elif any([n.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for n in
train.maker.fgraph.toposort()]):
    print('Used the gpu')
else:
    print('ERROR, not able to tell if theano used the cpu or the gpu')
    print(train.maker.fgraph.toposort())

for i in range(training_steps):
    pred, err = train()
#print "Final model:"
#print w.get_value(), b.get_value()

print("target values for D")
print(D[1])

print("prediction on D")
print(predict())

"""

# 2. Profiling


# 2.1 Profiling for CPU computations

# In your terminal, type:
$ THEANO_FLAGS=profile=True,device=cpu python using_gpu_solution_1.py

# You'll see first the output of the script:
Used the cpu
target values for D
prediction on D

# Followed by the output of profiling.. You'll see profiling results for each function
# in the script, followed by a summary for all functions.
# We'll show here only the summary:

Results were produced using an Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz

Function profiling
==================
  Message: Sum of all(2) printed profiles at exit excluding Scan op profile.
  Time in 10001 calls to Function.__call__: 1.300452e+00s
  Time in Function.fn.__call__: 1.215823e+00s (93.492%)
  Time in thunks: 1.157602e+00s (89.015%)
  Total compile time: 8.922548e-01s
    Number of Apply nodes: 17
    Theano Optimizer time: 6.270301e-01s
       Theano validate time: 5.993605e-03s
    Theano Linker time (includes C, CUDA code generation/compiling): 2.949309e-02s
       Import time 3.543139e-03s

Time in all call to theano.grad() 1.848292e-02s
Time since theano import 2.864s
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  64.5%    64.5%       0.747s       3.73e-05s     C    20001       3   theano.tensor.blas_c.CGemv
  33.1%    97.7%       0.384s       4.79e-06s     C    80001       9   theano.tensor.elemwise.Elemwise
   1.0%    98.6%       0.011s       1.14e-06s     C    10000       1   theano.tensor.elemwise.Sum
   0.7%    99.4%       0.009s       2.85e-07s     C    30001       4   theano.tensor.elemwise.DimShuffle
   0.3%    99.7%       0.004s       3.64e-07s     C    10001       2   theano.tensor.basic.AllocEmpty
   0.3%   100.0%       0.004s       1.78e-07s     C    20001       3   theano.compile.ops.Shape_i
   ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  64.5%    64.5%       0.747s       3.73e-05s     C     20001        3   CGemv{inplace}
  18.7%    83.2%       0.217s       2.17e-05s     C     10000        1   Elemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}[(0, 4)]
   8.9%    92.1%       0.103s       1.03e-05s     C     10000        1   Elemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((scalar_sigmoid((-i0)) * i1 * i4) / i3))}}[(0, 0)]
   4.3%    96.4%       0.050s       4.98e-06s     C     10000        1   Elemwise{Composite{GT(scalar_sigmoid(i0), i1)}}
   1.0%    97.4%       0.011s       1.14e-06s     C     10000        1   Sum{acc_dtype=float64}
   0.5%    97.9%       0.006s       2.83e-07s     C     20001        3   InplaceDimShuffle{x}
   0.4%    98.3%       0.004s       4.22e-07s     C     10000        1   Elemwise{sub,no_inplace}
   0.3%    98.6%       0.004s       3.70e-07s     C     10000        1   Elemwise{neg,no_inplace}
   0.3%    98.9%       0.004s       3.64e-07s     C     10001        2   AllocEmpty{dtype='float32'}
   0.3%    99.2%       0.004s       1.78e-07s     C     20001        3   Shape_i{0}
   0.2%    99.5%       0.003s       2.88e-07s     C     10000        1   InplaceDimShuffle{1,0}
   0.2%    99.7%       0.003s       2.65e-07s     C     10000        1   Elemwise{Composite{((-i0) - i1)}}[(0, 0)]
   0.2%    99.9%       0.002s       1.98e-07s     C     10000        1   Elemwise{Cast{float32}}
   0.1%   100.0%       0.002s       1.54e-07s     C     10000        1   Elemwise{Composite{(i0 - (i1 * i2))}}[(0, 0)]
   0.0%   100.0%       0.000s       4.77e-06s     C        1        1   Elemwise{Composite{GT(scalar_sigmoid((-((-i0) - i1))), i2)}}
   ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  34.0%    34.0%       0.394s       3.94e-05s   10000     7   CGemv{inplace}(AllocEmpty{dtype='float32'}.0, TensorConstant{1.0}, x, w, TensorConstant{0.0})
  30.5%    64.5%       0.353s       3.53e-05s   10000    15   CGemv{inplace}(w, TensorConstant{-0.00999999977648}, x.T, Elemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((scalar_sigmoid((-i0)) * i1 * i4) / i3))}}[(0, 0)].0, TensorConstant{0.999800026417})
  18.7%    83.2%       0.217s       2.17e-05s   10000    12   Elemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}[(0, 4)](y, Elemwise{Composite{((-i0) - i1)}}[(0, 0)].0, TensorConstant{(1,) of -1.0}, Elemwise{sub,no_inplace}.0, Elemwise{neg,no_inplace}.0)
   8.9%    92.1%       0.103s       1.03e-05s   10000    13   Elemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((scalar_sigmoid((-i0)) * i1 * i4) / i3))}}[(0, 0)](Elemwise{Composite{((-i0) - i1)}}[(0, 0)].0, TensorConstant{(1,) of -1.0}, y, Elemwise{Cast{float32}}.0, Elemwise{sub,no_inplace}.0)
   4.3%    96.4%       0.050s       4.98e-06s   10000    11   Elemwise{Composite{GT(scalar_sigmoid(i0), i1)}}(Elemwise{neg,no_inplace}.0, TensorConstant{(1,) of 0.5})
   1.0%    97.4%       0.011s       1.14e-06s   10000    14   Sum{acc_dtype=float64}(Elemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((scalar_sigmoid((-i0)) * i1 * i4) / i3))}}[(0, 0)].0)
   0.4%    97.8%       0.004s       4.22e-07s   10000     4   Elemwise{sub,no_inplace}(TensorConstant{(1,) of 1.0}, y)
   0.3%    98.1%       0.004s       3.76e-07s   10000     0   InplaceDimShuffle{x}(b)
   0.3%    98.4%       0.004s       3.70e-07s   10000    10   Elemwise{neg,no_inplace}(Elemwise{Composite{((-i0) - i1)}}[(0, 0)].0)
   0.3%    98.7%       0.004s       3.64e-07s   10000     5   AllocEmpty{dtype='float32'}(Shape_i{0}.0)
   0.2%    99.0%       0.003s       2.88e-07s   10000     2   InplaceDimShuffle{1,0}(x)
   0.2%    99.2%       0.003s       2.65e-07s   10000     9   Elemwise{Composite{((-i0) - i1)}}[(0, 0)](CGemv{inplace}.0, InplaceDimShuffle{x}.0)
   0.2%    99.4%       0.002s       2.21e-07s   10000     1   Shape_i{0}(x)
   0.2%    99.6%       0.002s       1.98e-07s   10000     8   Elemwise{Cast{float32}}(InplaceDimShuffle{x}.0)
   0.2%    99.7%       0.002s       1.90e-07s   10000     6   InplaceDimShuffle{x}(Shape_i{0}.0)
   0.1%    99.9%       0.002s       1.54e-07s   10000    16   Elemwise{Composite{(i0 - (i1 * i2))}}[(0, 0)](b, TensorConstant{0.00999999977648}, Sum{acc_dtype=float64}.0)
   0.1%   100.0%       0.001s       1.34e-07s   10000     3   Shape_i{0}(y)
   0.0%   100.0%       0.000s       3.89e-05s      1     3   CGemv{inplace}(AllocEmpty{dtype='float32'}.0, TensorConstant{1.0}, x, w, TensorConstant{0.0})
   0.0%   100.0%       0.000s       4.77e-06s      1     4   Elemwise{Composite{GT(scalar_sigmoid((-((-i0) - i1))), i2)}}(CGemv{inplace}.0, InplaceDimShuffle{x}.0, TensorConstant{(1,) of 0.5})
   0.0%   100.0%       0.000s       1.19e-06s      1     0   InplaceDimShuffle{x}(b)
   ... (remaining 2 Apply instances account for 0.00%(0.00s) of the runtime)




# 2.2 Profiling for GPU computations

# In your terminal, type:
$ CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS=profile=True,device=cuda python using_gpu_solution_1.py

# You'll see first the output of the script:
Used the gpu
target values for D
prediction on D

Results were produced using a GeForce GTX TITAN X

# Profiling summary for all functions:

Function profiling
==================
  Message: Sum of all(2) printed profiles at exit excluding Scan op profile.
  Time in 10001 calls to Function.__call__: 4.181247e+00s
  Time in Function.fn.__call__: 4.081113e+00s (97.605%)
  Time in thunks: 3.915566e+00s (93.646%)
  Total compile time: 9.256095e+00s
    Number of Apply nodes: 21
    Theano Optimizer time: 9.996419e-01s
       Theano validate time: 6.523132e-03s
    Theano Linker time (includes C, CUDA code generation/compiling): 8.239602e+00s
       Import time 4.228115e-03s

Time in all call to theano.grad() 3.286195e-02s
Time since theano import 15.415s
Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  59.5%    59.5%       2.329s       1.16e-04s     C    20001       3   theano.sandbox.gpuarray.blas.GpuGemv
  29.8%    89.3%       1.166s       1.30e-05s     C    90001      10   theano.sandbox.gpuarray.elemwise.GpuElemwise
   4.1%    93.4%       0.162s       8.10e-06s     C    20001       3   theano.sandbox.gpuarray.basic_ops.HostFromGpu
   3.3%    96.7%       0.131s       1.31e-05s     C    10000       1   theano.sandbox.gpuarray.elemwise.GpuCAReduceCuda
   1.6%    98.3%       0.061s       6.10e-06s     C    10000       1   theano.sandbox.gpuarray.basic_ops.GpuFromHost
   0.8%    99.1%       0.033s       1.09e-06s     C    30001       4   theano.sandbox.gpuarray.elemwise.GpuDimShuffle
   0.7%    99.8%       0.026s       2.59e-06s     C    10001       2   theano.sandbox.gpuarray.basic_ops.GpuAllocEmpty
   0.2%   100.0%       0.008s       3.95e-07s     C    20001       3   theano.compile.ops.Shape_i
   ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  59.5%    59.5%       2.329s       1.16e-04s     C     20001        3   GpuGemv{inplace=True}
   4.1%    63.6%       0.162s       8.10e-06s     C     20001        3   HostFromGpu(gpuarray)
   4.0%    67.6%       0.157s       1.57e-05s     C     10000        1   GpuElemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}[]<gpuarray>
   3.8%    71.4%       0.149s       1.49e-05s     C     10000        1   GpuElemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)]<gpuarray>
   3.7%    75.1%       0.144s       1.44e-05s     C     10000        1   GpuElemwise{sub,no_inplace}
   3.6%    78.7%       0.141s       1.41e-05s     C     10000        1   GpuElemwise{gt,no_inplace}
   3.4%    82.1%       0.133s       1.33e-05s     C     10000        1   GpuElemwise{Cast{float32}}[]<gpuarray>
   3.4%    85.5%       0.133s       1.33e-05s     C     10000        1   GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>
   3.3%    88.8%       0.131s       1.31e-05s     C     10000        1   GpuCAReduceCuda{add}
   2.9%    91.7%       0.112s       1.12e-05s     C     10000        1   GpuElemwise{neg,no_inplace}
   2.6%    94.3%       0.102s       1.02e-05s     C     10000        1   GpuElemwise{Composite{(i0 - (i1 * i2))}}[(0, 0)]<gpuarray>
   2.5%    96.7%       0.096s       9.63e-06s     C     10000        1   GpuElemwise{ScalarSigmoid}[(0, 0)]<gpuarray>
   1.6%    98.3%       0.061s       6.10e-06s     C     10000        1   GpuFromHost<None>
   0.7%    99.0%       0.026s       2.59e-06s     C     10001        2   GpuAllocEmpty{dtype='float32', context_name=None}
   0.5%    99.5%       0.021s       1.06e-06s     C     20001        3   InplaceGpuDimShuffle{x}
   0.3%    99.8%       0.011s       1.14e-06s     C     10000        1   InplaceGpuDimShuffle{1,0}
   0.2%   100.0%       0.008s       3.95e-07s     C     20001        3   Shape_i{0}
   0.0%   100.0%       0.000s       2.00e-05s     C        1        1   GpuElemwise{Composite{GT(scalar_sigmoid((-((-i0) - i1))), i2)}}[]<gpuarray>
   ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  55.0%    55.0%       2.154s       2.15e-04s   10000     7   GpuGemv{inplace=True}(GpuAllocEmpty{dtype='float32', context_name=None}.0, TensorConstant{1.0}, x, w, TensorConstant{0.0})
   4.5%    59.5%       0.176s       1.76e-05s   10000    18   GpuGemv{inplace=True}(w, TensorConstant{-0.00999999977648}, InplaceGpuDimShuffle{1,0}.0, GpuElemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)]<gpuarray>.0, TensorConstant{0.999800026417})
   4.0%    63.5%       0.157s       1.57e-05s   10000    12   GpuElemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}[]<gpuarray>(y, GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>.0, GpuArrayConstant{[-1.]}, GpuElemwise{sub,no_inplace}.0, GpuElemwise{neg,no_inplace}.0)
   3.8%    67.3%       0.149s       1.49e-05s   10000    15   GpuElemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)]<gpuarray>(GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>.0, GpuArrayConstant{[-1.]}, y, GpuElemwise{Cast{float32}}[]<gpuarray>.0, GpuElemwise{ScalarSigmoid}[(0, 0)]<gpuarray>.0, GpuElemwise{sub,no_inplace}.0)
   3.7%    71.0%       0.144s       1.44e-05s   10000     4   GpuElemwise{sub,no_inplace}(GpuArrayConstant{[ 1.]}, y)
   3.6%    74.6%       0.141s       1.41e-05s   10000    16   GpuElemwise{gt,no_inplace}(GpuElemwise{ScalarSigmoid}[(0, 0)]<gpuarray>.0, GpuArrayConstant{[ 0.5]})
   3.4%    78.0%       0.133s       1.33e-05s   10000    10   GpuElemwise{Cast{float32}}[]<gpuarray>(InplaceGpuDimShuffle{x}.0)
   3.4%    81.4%       0.133s       1.33e-05s   10000     9   GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>(GpuGemv{inplace=True}.0, InplaceGpuDimShuffle{x}.0)
   3.3%    84.7%       0.131s       1.31e-05s   10000    17   GpuCAReduceCuda{add}(GpuElemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)]<gpuarray>.0)
   2.9%    87.5%       0.112s       1.12e-05s   10000    11   GpuElemwise{neg,no_inplace}(GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>.0)
   2.6%    90.1%       0.102s       1.02e-05s   10000    20   GpuElemwise{Composite{(i0 - (i1 * i2))}}[(0, 0)]<gpuarray>(b, GpuArrayConstant{0.00999999977648}, GpuCAReduceCuda{add}.0)
   2.5%    92.6%       0.096s       9.63e-06s   10000    13   GpuElemwise{ScalarSigmoid}[(0, 0)]<gpuarray>(GpuElemwise{neg,no_inplace}.0)
   2.3%    94.9%       0.090s       9.04e-06s   10000    19   HostFromGpu(gpuarray)(GpuElemwise{gt,no_inplace}.0)
   1.8%    96.7%       0.072s       7.16e-06s   10000    14   HostFromGpu(gpuarray)(GpuElemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}[]<gpuarray>.0)
   1.6%    98.3%       0.061s       6.10e-06s   10000     6   GpuFromHost<None>(Shape_i{0}.0)
   0.7%    99.0%       0.026s       2.59e-06s   10000     5   GpuAllocEmpty{dtype='float32', context_name=None}(Shape_i{0}.0)
   0.3%    99.3%       0.013s       1.33e-06s   10000     0   InplaceGpuDimShuffle{x}(b)
   0.3%    99.6%       0.011s       1.14e-06s   10000     2   InplaceGpuDimShuffle{1,0}(x)
   0.2%    99.8%       0.008s       7.94e-07s   10000     8   InplaceGpuDimShuffle{x}(GpuFromHost<None>.0)
   0.1%    99.9%       0.005s       5.27e-07s   10000     1   Shape_i{0}(x)
   ... (remaining 7 Apply instances account for 0.07%(0.00s) of the runtime)


# 3. Conclusions

Examine and compare 'Ops' summaries for CPU and GPU. Usually GPU ops 'GpuFromHost' and 'HostFromGpu' by themselves
consume a large amount of extra time, but by making as few as possible data transfers between GPU and CPU, you can minimize their overhead.
Notice that each of the GPU ops consumes more time than its CPU counterpart. This is because the ops operate on small inputs;
if you increase the input data size (e.g. set N = 4000), you will see a gain from using the GPU.

"""
