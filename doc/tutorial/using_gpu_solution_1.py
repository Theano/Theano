#!/usr/bin/env python
# Theano tutorial
# Solution to Exercise in section 'Using the GPU'


# 1. Raw results


from __future__ import absolute_import, print_function, division
import numpy
import theano
import theano.tensor as tt

from theano import sandbox, Out

theano.config.floatX = 'float32'

rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats).astype(theano.config.floatX),
rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
training_steps = 10000

# Declare Theano symbolic variables
x = theano.shared(D[0], name="x")
y = theano.shared(D[1], name="y")
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

"""
# Compile expressions to functions
train = theano.function(
            inputs=[x, y],
            outputs=[Out(theano.sandbox.cuda.basic_ops.gpu_from_host(tt.cast(prediction, 'float32')),borrow=True), Out(theano.sandbox.cuda.basic_ops.gpu_from_host(tt.cast(xent, 'float32')), borrow=True)],
            updates={w: w - 0.01 * gw, b: b - 0.01 * gb},
            name="train")
predict = theano.function(inputs=[x], outputs=Out(theano.sandbox.cuda.basic_ops.gpu_from_host(tt.cast(prediction, 'float32')), borrow=True),
            name="predict")
"""

# Compile expressions to functions
train = theano.function(
            inputs=[],
            outputs=[prediction, xent],
            updates={w: w - 0.01 * gw, b: b - 0.01 * gb},
            name="train")
predict = theano.function(inputs=[], outputs=prediction,
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

Results were produced using an Intel(R) Core(TM) i7-4820K CPU @ 3.70GHz

Function profiling
==================
  Message: Sum of all(3) printed profiles at exit excluding Scan op profile.
  Time in 10002 calls to Function.__call__: 1.590916e+00s
  Time in Function.fn.__call__: 1.492365e+00s (93.805%)
  Time in thunks: 1.408159e+00s (88.512%)
  Total compile time: 6.309664e+00s
    Number of Apply nodes: 25
    Theano Optimizer time: 4.848340e-01s
       Theano validate time: 5.454302e-03s
    Theano Linker time (includes C, CUDA code generation/compiling): 5.691789e+00s

Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  59.6%    59.6%       0.839s       4.19e-05s     C    20001       3   theano.tensor.blas_c.CGemv
  30.1%    89.7%       0.424s       4.71e-06s     C    90001      10   theano.tensor.elemwise.Elemwise
   5.5%    95.2%       0.078s       7.79e-02s     Py       1       1   theano.tensor.blas.Gemv
   1.9%    97.1%       0.026s       1.30e-06s     C    20001       3   theano.tensor.basic.Alloc
   1.3%    98.4%       0.018s       1.85e-06s     C    10000       1   theano.tensor.elemwise.Sum
   1.0%    99.4%       0.014s       4.78e-07s     C    30001       4   theano.tensor.elemwise.DimShuffle
   0.6%   100.0%       0.008s       4.23e-07s     C    20001       3   theano.compile.ops.Shape_i
   ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  59.6%    59.6%       0.839s       4.19e-05s     C     20001        3   CGemv{inplace}
  15.8%    75.4%       0.223s       2.23e-05s     C     10000        1   Elemwise{Composite{[sub(mul(i0, scalar_softplus(i1)), mul(i2, i3, scalar_softplus(i4)))]}}[(0, 4)]
   7.7%    83.1%       0.109s       1.09e-05s     C     10000        1   Elemwise{Composite{[add(mul(scalar_sigmoid(i0), i1, i2, i3), true_div(mul(scalar_sigmoid(neg(i0)), i4), i5))]}}[(0, 0)]
   5.5%    88.7%       0.078s       7.79e-02s     Py       1        1   Gemv{no_inplace}
   4.3%    92.9%       0.060s       6.00e-06s     C     10000        1   Elemwise{Composite{[GT(scalar_sigmoid(i0), i1)]}}
   1.9%    94.8%       0.026s       1.30e-06s     C     20001        3   Alloc
   1.3%    96.1%       0.018s       1.85e-06s     C     10000        1   Sum{acc_dtype=float64}
   0.7%    96.8%       0.009s       4.73e-07s     C     20001        3   InplaceDimShuffle{x}
   0.6%    97.4%       0.009s       8.52e-07s     C     10000        1   Elemwise{sub,no_inplace}
   0.6%    98.0%       0.008s       4.23e-07s     C     20001        3   Shape_i{0}
   0.5%    98.5%       0.007s       7.06e-07s     C     10000        1   Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)]
   0.5%    98.9%       0.007s       6.57e-07s     C     10000        1   Elemwise{neg,no_inplace}
   0.3%    99.3%       0.005s       4.88e-07s     C     10000        1   InplaceDimShuffle{1,0}
   0.3%    99.5%       0.004s       3.78e-07s     C     10000        1   Elemwise{inv,no_inplace}
   0.2%    99.8%       0.003s       3.44e-07s     C     10000        1   Elemwise{Cast{float32}}
   0.2%   100.0%       0.003s       3.01e-07s     C     10000        1   Elemwise{Composite{[sub(i0, mul(i1, i2))]}}[(0, 0)]
   0.0%   100.0%       0.000s       8.11e-06s     C        1        1   Elemwise{Composite{[GT(scalar_sigmoid(neg(sub(neg(i0), i1))), i2)]}}
   ... (remaining 0 Ops account for   0.00%(0.00s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
  31.6%    31.6%       0.445s       4.45e-05s   10000     7   CGemv{inplace}(Alloc.0, TensorConstant{1.0}, x, w, TensorConstant{0.0})
  27.9%    59.6%       0.393s       3.93e-05s   10000    17   CGemv{inplace}(w, TensorConstant{-0.00999999977648}, x.T, Elemwise{Composite{[add(mul(scalar_sigmoid(i0), i1, i2, i3), true_div(mul(scalar_sigmoid(neg(i0)), i4), i5))]}}[(0, 0)].0, TensorConstant{0.999800026417})
  15.8%    75.4%       0.223s       2.23e-05s   10000    14   Elemwise{Composite{[sub(mul(i0, scalar_softplus(i1)), mul(i2, i3, scalar_softplus(i4)))]}}[(0, 4)](y, Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0, TensorConstant{(1,) of -1.0}, Elemwise{sub,no_inplace}.0, Elemwise{neg,no_inplace}.0)
   7.7%    83.1%       0.109s       1.09e-05s   10000    15   Elemwise{Composite{[add(mul(scalar_sigmoid(i0), i1, i2, i3), true_div(mul(scalar_sigmoid(neg(i0)), i4), i5))]}}[(0, 0)](Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0, TensorConstant{(1,) of -1.0}, Alloc.0, y, Elemwise{sub,no_inplace}.0, Elemwise{Cast{float32}}.0)
   5.5%    88.7%       0.078s       7.79e-02s      1     0   Gemv{no_inplace}(aa, TensorConstant{1.0}, xx, yy, TensorConstant{0.0})
   4.3%    92.9%       0.060s       6.00e-06s   10000    13   Elemwise{Composite{[GT(scalar_sigmoid(i0), i1)]}}(Elemwise{neg,no_inplace}.0, TensorConstant{(1,) of 0.5})
   1.3%    94.2%       0.018s       1.85e-06s   10000    16   Sum{acc_dtype=float64}(Elemwise{Composite{[add(mul(scalar_sigmoid(i0), i1, i2, i3), true_div(mul(scalar_sigmoid(neg(i0)), i4), i5))]}}[(0, 0)].0)
   1.0%    95.2%       0.013s       1.34e-06s   10000     5   Alloc(TensorConstant{0.0}, Shape_i{0}.0)
   0.9%    96.1%       0.013s       1.27e-06s   10000    12   Alloc(Elemwise{inv,no_inplace}.0, Shape_i{0}.0)
   0.6%    96.7%       0.009s       8.52e-07s   10000     4   Elemwise{sub,no_inplace}(TensorConstant{(1,) of 1.0}, y)
   0.5%    97.2%       0.007s       7.06e-07s   10000     9   Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)](CGemv{inplace}.0, InplaceDimShuffle{x}.0)
   0.5%    97.6%       0.007s       6.57e-07s   10000    11   Elemwise{neg,no_inplace}(Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0)
   0.4%    98.1%       0.006s       6.27e-07s   10000     0   InplaceDimShuffle{x}(b)
   0.4%    98.5%       0.006s       5.90e-07s   10000     1   Shape_i{0}(x)
   0.3%    98.9%       0.005s       4.88e-07s   10000     2   InplaceDimShuffle{1,0}(x)
   0.3%    99.1%       0.004s       3.78e-07s   10000    10   Elemwise{inv,no_inplace}(Elemwise{Cast{float32}}.0)
   0.2%    99.4%       0.003s       3.44e-07s   10000     8   Elemwise{Cast{float32}}(InplaceDimShuffle{x}.0)
   0.2%    99.6%       0.003s       3.19e-07s   10000     6   InplaceDimShuffle{x}(Shape_i{0}.0)
   0.2%    99.8%       0.003s       3.01e-07s   10000    18   Elemwise{Composite{[sub(i0, mul(i1, i2))]}}[(0, 0)](b, TensorConstant{0.00999999977648}, Sum{acc_dtype=float64}.0)
   0.2%   100.0%       0.003s       2.56e-07s   10000     3   Shape_i{0}(y)
   ... (remaining 5 Apply instances account for 0.00%(0.00s) of the runtime)



# 2.2 Profiling for GPU computations

# In your terminal, type:
$ CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS=profile=True,device=gpu python using_gpu_solution_1.py

# You'll see first the output of the script:
Used the gpu
target values for D
prediction on D

Results were produced using a GeForce GTX TITAN

# Profiling summary for all functions:

Function profiling
==================
  Message: Sum of all(3) printed profiles at exit excluding Scan op profile.
  Time in 10002 calls to Function.__call__: 3.535239e+00s
  Time in Function.fn.__call__: 3.420863e+00s (96.765%)
  Time in thunks: 2.865905e+00s (81.067%)
  Total compile time: 4.728150e-01s
    Number of Apply nodes: 36
    Theano Optimizer time: 4.283385e-01s
       Theano validate time: 7.687330e-03s
    Theano Linker time (includes C, CUDA code generation/compiling): 2.801418e-02s

Class
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
  45.7%    45.7%       1.308s       1.64e-05s     C    80001       9   theano.sandbox.cuda.basic_ops.GpuElemwise
  17.2%    62.8%       0.492s       2.46e-05s     C    20002       4   theano.sandbox.cuda.blas.GpuGemv
  15.1%    77.9%       0.433s       2.17e-05s     C    20001       3   theano.sandbox.cuda.basic_ops.GpuAlloc
   8.2%    86.1%       0.234s       1.17e-05s     C    20002       4   theano.sandbox.cuda.basic_ops.HostFromGpu
   7.2%    93.3%       0.207s       2.07e-05s     C    10000       1   theano.sandbox.cuda.basic_ops.GpuCAReduce
   4.4%    97.7%       0.127s       1.27e-05s     C    10003       4   theano.sandbox.cuda.basic_ops.GpuFromHost
   0.9%    98.6%       0.025s       8.23e-07s     C    30001       4   theano.sandbox.cuda.basic_ops.GpuDimShuffle
   0.7%    99.3%       0.020s       9.88e-07s     C    20001       3   theano.tensor.elemwise.Elemwise
   0.5%    99.8%       0.014s       7.18e-07s     C    20001       3   theano.compile.ops.Shape_i
   0.2%   100.0%       0.006s       5.78e-07s     C    10000       1   theano.tensor.elemwise.DimShuffle
   ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)

Ops
---
<% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
  17.2%    17.2%       0.492s       2.46e-05s     C     20001        3   GpuGemv{inplace}
   8.2%    25.3%       0.234s       1.17e-05s     C     20002        4   HostFromGpu
   8.0%    33.3%       0.228s       2.28e-05s     C     10001        2   GpuAlloc{memset_0=True}
   7.4%    40.7%       0.211s       2.11e-05s     C     10000        1   GpuElemwise{Composite{[sub(mul(i0, scalar_softplus(i1)), mul(i2, i3, scalar_softplus(i4)))]},no_inplace}
   7.2%    47.9%       0.207s       2.07e-05s     C     10000        1   GpuCAReduce{add}{1}
   7.1%    55.0%       0.205s       2.05e-05s     C     10000        1   GpuAlloc
   6.9%    62.0%       0.198s       1.98e-05s     C     10000        1   GpuElemwise{sub,no_inplace}
   6.9%    68.9%       0.198s       1.98e-05s     C     10000        1   GpuElemwise{inv,no_inplace}
   6.2%    75.1%       0.178s       1.78e-05s     C     10000        1   GpuElemwise{neg,no_inplace}
   5.6%    80.6%       0.159s       1.59e-05s     C     10000        1   GpuElemwise{Composite{[add(mul(scalar_sigmoid(i0), i1, i2, i3), true_div(mul(i4, i5), i6))]}}[(0, 0)]
   4.4%    85.1%       0.127s       1.27e-05s     C     10003        4   GpuFromHost
   4.3%    89.4%       0.124s       1.24e-05s     C     10000        1   GpuElemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)]
   4.2%    93.6%       0.121s       1.21e-05s     C     10000        1   GpuElemwise{ScalarSigmoid}[(0, 0)]
   4.2%    97.7%       0.119s       1.19e-05s     C     10000        1   GpuElemwise{Composite{[sub(i0, mul(i1, i2))]}}[(0, 0)]
   0.5%    98.2%       0.014s       7.18e-07s     C     20001        3   Shape_i{0}
   0.5%    98.7%       0.013s       1.33e-06s     C     10001        2   Elemwise{gt,no_inplace}
   0.3%    99.0%       0.010s       9.81e-07s     C     10000        1   GpuDimShuffle{1,0}
   0.3%    99.3%       0.008s       7.90e-07s     C     10000        1   GpuDimShuffle{0}
   0.2%    99.6%       0.007s       6.97e-07s     C     10001        2   GpuDimShuffle{x}
   0.2%    99.8%       0.006s       6.50e-07s     C     10000        1   Elemwise{Cast{float32}}
   ... (remaining 3 Ops account for   0.20%(0.01s) of the runtime)

Apply
------
<% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
   8.8%     8.8%       0.251s       2.51e-05s   10000    22   GpuGemv{inplace}(w, TensorConstant{-0.00999999977648}, GpuDimShuffle{1,0}.0, GpuElemwise{Composite{[add(mul(scalar_sigmoid(i0), i1, i2, i3), true_div(mul(i4, i5), i6))]}}[(0, 0)].0, TensorConstant{0.999800026417})
   8.4%    17.2%       0.241s       2.41e-05s   10000     7   GpuGemv{inplace}(GpuAlloc{memset_0=True}.0, TensorConstant{1.0}, x, w, TensorConstant{0.0})
   8.0%    25.1%       0.228s       2.28e-05s   10000     5   GpuAlloc{memset_0=True}(CudaNdarrayConstant{[ 0.]}, Shape_i{0}.0)
   7.4%    32.5%       0.211s       2.11e-05s   10000    13   GpuElemwise{Composite{[sub(mul(i0, scalar_softplus(i1)), mul(i2, i3, scalar_softplus(i4)))]},no_inplace}(y, GpuElemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0, CudaNdarrayConstant{[-1.]}, GpuElemwise{sub,no_inplace}.0, GpuElemwise{neg,no_inplace}.0)
   7.2%    39.7%       0.207s       2.07e-05s   10000    21   GpuCAReduce{add}{1}(GpuElemwise{Composite{[add(mul(scalar_sigmoid(i0), i1, i2, i3), true_div(mul(i4, i5), i6))]}}[(0, 0)].0)
   7.1%    46.9%       0.205s       2.05e-05s   10000    17   GpuAlloc(GpuDimShuffle{0}.0, Shape_i{0}.0)
   6.9%    53.8%       0.198s       1.98e-05s   10000     4   GpuElemwise{sub,no_inplace}(CudaNdarrayConstant{[ 1.]}, y)
   6.9%    60.7%       0.198s       1.98e-05s   10000    12   GpuElemwise{inv,no_inplace}(GpuFromHost.0)
   6.2%    66.9%       0.178s       1.78e-05s   10000    11   GpuElemwise{neg,no_inplace}(GpuElemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0)
   5.6%    72.5%       0.159s       1.59e-05s   10000    19   GpuElemwise{Composite{[add(mul(scalar_sigmoid(i0), i1, i2, i3), true_div(mul(i4, i5), i6))]}}[(0, 0)](GpuElemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0, CudaNdarrayConstant{[-1.]}, GpuAlloc.0, y, GpuElemwise{ScalarSigmoid}[(0, 0)].0, GpuElemwise{sub,no_inplace}.0, GpuFromHost.0)
   4.8%    77.3%       0.138s       1.38e-05s   10000    18   HostFromGpu(GpuElemwise{ScalarSigmoid}[(0, 0)].0)
   4.4%    81.7%       0.126s       1.26e-05s   10000    10   GpuFromHost(Elemwise{Cast{float32}}.0)
   4.3%    86.0%       0.124s       1.24e-05s   10000     9   GpuElemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)](GpuGemv{inplace}.0, GpuDimShuffle{x}.0)
   4.2%    90.2%       0.121s       1.21e-05s   10000    15   GpuElemwise{ScalarSigmoid}[(0, 0)](GpuElemwise{neg,no_inplace}.0)
   4.2%    94.4%       0.119s       1.19e-05s   10000    23   GpuElemwise{Composite{[sub(i0, mul(i1, i2))]}}[(0, 0)](b, CudaNdarrayConstant{0.00999999977648}, GpuCAReduce{add}{1}.0)
   3.4%    97.7%       0.096s       9.61e-06s   10000    16   HostFromGpu(GpuElemwise{Composite{[sub(mul(i0, scalar_softplus(i1)), mul(i2, i3, scalar_softplus(i4)))]},no_inplace}.0)
   0.5%    98.2%       0.013s       1.33e-06s   10000    20   Elemwise{gt,no_inplace}(HostFromGpu.0, TensorConstant{(1,) of 0.5})
   0.3%    98.5%       0.010s       9.81e-07s   10000     2   GpuDimShuffle{1,0}(x)
   0.3%    98.8%       0.008s       8.27e-07s   10000     1   Shape_i{0}(x)
   0.3%    99.1%       0.008s       7.90e-07s   10000    14   GpuDimShuffle{0}(GpuElemwise{inv,no_inplace}.0)
   ... (remaining 16 Apply instances account for 0.90%(0.03s) of the runtime)


# 3. Conclusions

Examine and compare 'Ops' summaries for CPU and GPU. Usually GPU ops 'GpuFromHost' and 'HostFromGpu' by themselves
consume a large amount of extra time, but by making as few as possible data transfers between GPU and CPU, you can minimize their overhead.
Notice that each of the GPU ops consumes more time than its CPU counterpart. This is because the ops operate on small inputs;
if you increase the input data size (e.g. set N = 4000), you will see a gain from using the GPU.

"""
