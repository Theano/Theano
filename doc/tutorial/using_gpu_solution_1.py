#!/usr/bin/env python
# Theano tutorial
# Solution to Exercise in section 'Using the GPU'


# 1. Raw results
#
# same code as in mode_solution_1 but run with following command lines:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu time python program_name.py
# THEANO_FLAGS=mode=FAST_RUN,device=cpu time python program_name.py
# for GPU and CPU respectively
# typical time: 20 sec (CPU), 10 sec (GPU)

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
    print 'Used the cpu'
elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
train.maker.fgraph.toposort()]):
    print 'Used the gpu'
else:
    print 'ERROR, not able to tell if theano used the cpu or the gpu'
    print train.maker.fgraph.toposort()

for i in range(training_steps):
    pred, err = train()
#print "Final model:"
#print w.get_value(), b.get_value()

print "target values for D"
print D[1]

print "prediction on D"
print predict()

"""

# 2. Profiling
#
# same code as above but run with following command lines:
# THEANO_FLAGS=mode=ProfileMode,device=gpu python program_name.py
# THEANO_FLAGS=mode=ProfileMode,device=cpu python program_name.py
# for GPU and CPU


# 2.1 Profiling output for CPU computations


$ THEANO_FLAGS=mode=ProfileMode,device=cpu python program_name.py
Used the cpu
target values for D
prediction on D
Used the cpu
target values for D
prediction on D

ProfileMode.print_summary()
---------------------------

Time since import 12.586s
Theano compile time: 0.000s (0.0% since import)
    Optimization time: 0.000s
    Linker time: 0.000s
Theano fct call 5.147s (40.9% since import)
   Theano Op time 3.595s 28.6%(since import) 69.8%(of fct call)
   Theano function overhead in ProfileMode 1.552s 12.3%(since import) 30.2%(of fct call)
20002 Theano fct call, 0.000s per call
Rest of the time since import 7.440s 59.1%

Theano fct summary:
<% total fct time> <total time> <time per call> <nb call> <fct name>
   49.9% 2.567s 2.57e-04s 10000 train
    0.0% 0.000s 1.24e-04s 1 predict
    0.0% 0.000s 1.26e-04s 1 predict
   50.1% 2.579s 2.58e-04s 10000 train

Single Op-wise summary:
<% of local_time spent on this kind of Op> <cumulative %> <self seconds> <cumulative seconds> <time per call> [*] <nb_call> <nb_op> <nb_apply> <Op name>
   59.3%   59.3%  2.133s  2.133s  5.33e-05s * 40002  1  6 <class 'theano.tensor.blas_c.CGemv'>
   34.4%   93.8%  1.238s  3.371s  6.19e-06s * 200002 11 22 <class 'theano.tensor.elemwise.Elemwise'>
    2.8%   96.6%  0.100s  3.471s  2.51e-06s * 40002  1  6 <class 'theano.tensor.basic.Alloc'>
    2.1%   98.7%  0.075s  3.546s  1.26e-06s * 60002  2  8 <class 'theano.tensor.elemwise.DimShuffle'>
    0.7%   99.3%  0.024s  3.571s  6.11e-07s * 40002  1  6 <class 'theano.tensor.opt.Shape_i'>
    0.7%  100.0%  0.024s  3.595s  1.18e-06s * 20000  1  2 <class 'theano.tensor.elemwise.Sum'>
   ... (remaining 0 single Op account for 0.00%(0.00s) of the runtime)
(*) Op is running a c implementation

Op-wise summary:
<% of local_time spent on this kind of Op> <cumulative %> <self seconds> <cumulative seconds> <time per call> [*]  <nb_call> <nb apply> <Op name>
   59.3%   59.3%  2.133s  2.133s  5.33e-05s * 40002  6 CGemv{inplace}
   18.1%   77.4%  0.650s  2.783s  3.25e-05s * 20000  2 Elemwise{Composite{[Composite{[Composite{[sub(mul(i0, i1), neg(i2))]}(i0, scalar_softplus(i1), mul(i2, i3))]}(i0, i1, i2, scalar_softplus(i3))]}}
    6.4%   83.9%  0.231s  3.014s  1.16e-05s * 20000  2 Elemwise{Composite{[Composite{[Composite{[Composite{[mul(i0, add(i1, i2))]}(i0, neg(i1), true_div(i2, i3))]}(i0, mul(i1, i2, i3), i4, i5)]}(i0, i1, i2, exp(i3), i4, i5)]}}[(0, 0)]
    4.0%   87.8%  0.142s  3.157s  7.11e-06s * 20000  2 Elemwise{ScalarSigmoid{output_types_preference=transfer_type{0}}}[(0, 0)]
    2.8%   90.6%  0.100s  3.257s  2.51e-06s * 40002  6 Alloc
    1.4%   92.1%  0.052s  3.309s  1.30e-06s * 40002  6 InplaceDimShuffle{x}
    1.1%   93.1%  0.038s  3.347s  1.92e-06s * 20000  2 Elemwise{Cast{float32}}
    1.1%   94.2%  0.038s  3.386s  1.91e-06s * 20000  2 Elemwise{sub,no_inplace}
    1.0%   95.2%  0.036s  3.421s  1.79e-06s * 20000  2 Elemwise{gt,no_inplace}
    0.8%   96.0%  0.029s  3.450s  1.44e-06s * 20000  2 Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)]
    0.8%   96.8%  0.028s  3.479s  1.42e-06s * 20000  2 Elemwise{neg,no_inplace}
    0.7%   97.5%  0.024s  3.503s  6.11e-07s * 40002  6 Shape_i{0}
    0.7%   98.1%  0.024s  3.527s  1.18e-06s * 20000  2 Sum
    0.6%   98.8%  0.023s  3.550s  1.16e-06s * 20000  2 InplaceDimShuffle{1,0}
    0.6%   99.4%  0.023s  3.573s  1.15e-06s * 20000  2 Elemwise{Composite{[sub(i0, mul(i1, i2))]}}[(0, 0)]
    0.6%  100.0%  0.022s  3.595s  1.08e-06s * 20000  2 Elemwise{inv,no_inplace}
    0.0%  100.0%  0.000s  3.595s  1.19e-05s *     2  2 Elemwise{Composite{[Composite{[Composite{[Composite{[GT(scalar_sigmoid(i0), i1)]}(neg(i0), i1)]}(sub(i0, i1), i2)]}(neg(i0), i1, i2)]}}
   ... (remaining 0 Op account for   0.00%(0.00s) of the runtime)
(*) Op is running a c implementation

Apply-wise summary:
<% of local_time spent at this position> <cumulative %%> <apply time> <cumulative seconds> <time per call> [*] <nb_call> <Apply position> <Apply Op name>
   14.9%   14.9%  0.536s  0.536s 5.36e-05s  * 10000   7 CGemv{inplace}(Alloc.0, TensorConstant{1.0}, x, w, TensorConstant{1.0})
   14.9%   29.8%  0.534s  1.070s 5.34e-05s  * 10000  18 CGemv{inplace}(w, TensorConstant{-0.00999999977648}, x.T, Elemwise{Composite{[Composite{[Composite{[Composite{[mul(i0, add(i1, i2))]}(i0, neg(i1), true_div(i2, i3))]}(i0, mul(i1, i2, i3), i4, i5)]}(i0, i1, i2, exp(i3), i4, i5)]}}[(0, 0)].0, TensorConstant{0.999800026417})
   14.8%   44.6%  0.532s  1.602s 5.32e-05s  * 10000   7 CGemv{inplace}(Alloc.0, TensorConstant{1.0}, x, w, TensorConstant{1.0})
   14.7%   59.3%  0.530s  2.132s 5.30e-05s  * 10000  18 CGemv{inplace}(w, TensorConstant{-0.00999999977648}, x.T, Elemwise{Composite{[Composite{[Composite{[Composite{[mul(i0, add(i1, i2))]}(i0, neg(i1), true_div(i2, i3))]}(i0, mul(i1, i2, i3), i4, i5)]}(i0, i1, i2, exp(i3), i4, i5)]}}[(0, 0)].0, TensorConstant{0.999800026417})
    9.1%   68.4%  0.327s  2.460s 3.27e-05s  * 10000  13 Elemwise{Composite{[Composite{[Composite{[sub(mul(i0, i1), neg(i2))]}(i0, scalar_softplus(i1), mul(i2, i3))]}(i0, i1, i2, scalar_softplus(i3))]}}(y, Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0, Elemwise{sub,no_inplace}.0, Elemwise{neg,no_inplace}.0)
    9.0%   77.4%  0.323s  2.783s 3.23e-05s  * 10000  13 Elemwise{Composite{[Composite{[Composite{[sub(mul(i0, i1), neg(i2))]}(i0, scalar_softplus(i1), mul(i2, i3))]}(i0, i1, i2, scalar_softplus(i3))]}}(y, Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0, Elemwise{sub,no_inplace}.0, Elemwise{neg,no_inplace}.0)
    3.2%   80.6%  0.116s  2.899s 1.16e-05s  * 10000  16 Elemwise{Composite{[Composite{[Composite{[Composite{[mul(i0, add(i1, i2))]}(i0, neg(i1), true_div(i2, i3))]}(i0, mul(i1, i2, i3), i4, i5)]}(i0, i1, i2, exp(i3), i4, i5)]}}[(0, 0)](Elemwise{ScalarSigmoid{output_types_preference=transfer_type{0}}}[(0, 0)].0, Alloc.0, y, Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0, Elemwise{sub,no_inplace}.0, Elemwise{Cast{float32}}.0)
    3.2%   83.9%  0.116s  3.014s 1.16e-05s  * 10000  16 Elemwise{Composite{[Composite{[Composite{[Composite{[mul(i0, add(i1, i2))]}(i0, neg(i1), true_div(i2, i3))]}(i0, mul(i1, i2, i3), i4, i5)]}(i0, i1, i2, exp(i3), i4, i5)]}}[(0, 0)](Elemwise{ScalarSigmoid{output_types_preference=transfer_type{0}}}[(0, 0)].0, Alloc.0, y, Elemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)].0, Elemwise{sub,no_inplace}.0, Elemwise{Cast{float32}}.0)
    2.0%   85.8%  0.071s  3.086s 7.12e-06s  * 10000  14 Elemwise{ScalarSigmoid{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{neg,no_inplace}.0)
    2.0%   87.8%  0.071s  3.156s 7.09e-06s  * 10000  14 Elemwise{ScalarSigmoid{output_types_preference=transfer_type{0}}}[(0, 0)](Elemwise{neg,no_inplace}.0)
    0.9%   88.8%  0.034s  3.190s 3.38e-06s  * 10000  12 Alloc(Elemwise{inv,no_inplace}.0, Shape_i{0}.0)
    0.9%   89.7%  0.034s  3.224s 3.37e-06s  * 10000  12 Alloc(Elemwise{inv,no_inplace}.0, Shape_i{0}.0)
    0.5%   90.2%  0.019s  3.243s 1.93e-06s  * 10000   8 Elemwise{Cast{float32}}(InplaceDimShuffle{x}.0)
    0.5%   90.8%  0.019s  3.262s 1.92e-06s  * 10000   4 Elemwise{sub,no_inplace}(TensorConstant{(1,) of 1.0}, y)
    0.5%   91.3%  0.019s  3.282s 1.90e-06s  * 10000   4 Elemwise{sub,no_inplace}(TensorConstant{(1,) of 1.0}, y)
   ... (remaining 35 Apply instances account for 8.71%(0.31s) of the runtime)
(*) Op is running a c implementation

Profile of Theano functions memory:
(This check only the output of each apply node. It don't check the temporary memory used by the op in the apply node.)
   We skipped 4 theano function(s). Each of them used less then 1024B(theano flags ProfileMode.min_memory_size) of total intermediate memory size

Here are tips to potentially make your code run faster
(if you think of new ones, suggest them on the mailing list).
Test them first, as they are not guaranteed to always provide a speedup.
  Sorry, no tip for today.


# 2.2 Profiling output for GPU computations

$ THEANO_FLAGS=mode=ProfileMode,device=gpu python program_name.py
Using gpu device 0: GeForce GTX 580
Used the gpu
target values for D
prediction on D
Used the gpu
target values for D
prediction on D

ProfileMode.print_summary()
---------------------------

Time since import 25.682s
Theano compile time: 0.000s (0.0% since import)
    Optimization time: 0.000s
    Linker time: 0.000s
Theano fct call 17.052s (66.4% since import)
   Theano Op time 14.548s 56.6%(since import) 85.3%(of fct call)
   Theano function overhead in ProfileMode 2.505s 9.8%(since import) 14.7%(of fct call)
20002 Theano fct call, 0.001s per call
Rest of the time since import 8.630s 33.6%

Theano fct summary:
<% total fct time> <total time> <time per call> <nb call> <fct name>
   50.0% 8.526s 8.53e-04s 10000 train
    0.0% 0.001s 1.09e-03s 1 predict
   50.0% 8.524s 8.52e-04s 10000 train
    0.0% 0.001s 1.10e-03s 1 predict

Single Op-wise summary:
<% of local_time spent on this kind of Op> <cumulative %> <self seconds> <cumulative seconds> <time per call> [*] <nb_call> <nb_op> <nb_apply> <Op name>
   54.8%   54.8%  7.968s  7.968s  1.33e-04s   60002  1  8 <class 'theano.sandbox.cuda.basic_ops.GpuFromHost'>
   16.2%   71.0%  2.358s  10.325s  1.47e-05s * 160002  9 18 <class 'theano.sandbox.cuda.basic_ops.GpuElemwise'>
   12.3%   83.3%  1.795s  12.120s  4.49e-05s * 40002  1  6 <class 'theano.sandbox.cuda.blas.GpuGemv'>
    7.0%   90.4%  1.024s  13.144s  2.56e-05s   40002  1  6 <class 'theano.sandbox.cuda.basic_ops.HostFromGpu'>
    5.0%   95.4%  0.728s  13.872s  1.82e-05s * 40002  1  6 <class 'theano.sandbox.cuda.basic_ops.GpuAlloc'>
    2.1%   97.4%  0.300s  14.171s  1.50e-05s * 20000  1  2 <class 'theano.sandbox.cuda.basic_ops.GpuSum'>
    1.3%   98.7%  0.189s  14.360s  3.15e-06s * 60002  3  8 <class 'theano.sandbox.cuda.basic_ops.GpuDimShuffle'>
    0.6%   99.4%  0.094s  14.454s  2.35e-06s * 40002  2  6 <class 'theano.tensor.elemwise.Elemwise'>
    0.3%   99.7%  0.048s  14.503s  1.21e-06s * 40002  1  6 <class 'theano.tensor.opt.Shape_i'>
    0.3%  100.0%  0.045s  14.548s  2.25e-06s * 20000  1  2 <class 'theano.tensor.elemwise.DimShuffle'>
   ... (remaining 0 single Op account for 0.00%(0.00s) of the runtime)
(*) Op is running a c implementation

Op-wise summary:
<% of local_time spent on this kind of Op> <cumulative %> <self seconds> <cumulative seconds> <time per call> [*]  <nb_call> <nb apply> <Op name>
   54.8%   54.8%  7.968s  7.968s  1.33e-04s   60002  8 GpuFromHost
   12.3%   67.1%  1.795s  9.763s  4.49e-05s * 40002  6 GpuGemv{inplace}
    7.0%   74.1%  1.024s  10.786s  2.56e-05s   40002  6 HostFromGpu
    5.0%   79.1%  0.728s  11.514s  1.82e-05s * 40002  6 GpuAlloc
    2.3%   81.4%  0.334s  11.848s  1.67e-05s * 20000  2 GpuElemwise{Composite{[Composite{[Composite{[Composite{[mul(i0, add(i1, i2))]}(i0, neg(i1), true_div(i2, i3))]}(i0, mul(i1, i2, i3), i4, i5)]}(i0, i1, i2, exp(i3), i4, i5)]}}[(0, 0)]
    2.2%   83.6%  0.319s  12.167s  1.59e-05s * 20000  2 GpuElemwise{Composite{[Composite{[Composite{[sub(mul(i0, i1), neg(i2))]}(i0, scalar_softplus(i1), mul(i2, i3))]}(i0, i1, i2, scalar_softplus(i3))]},no_inplace}
    2.1%   85.7%  0.301s  12.468s  1.50e-05s * 20000  2 GpuElemwise{neg,no_inplace}
    2.1%   87.8%  0.300s  12.768s  1.50e-05s * 20000  2 GpuSum{1}
    2.0%   89.8%  0.292s  13.060s  1.46e-05s * 20000  2 GpuElemwise{inv,no_inplace}
    1.9%   91.7%  0.283s  13.343s  1.42e-05s * 20000  2 GpuElemwise{Composite{[sub(neg(i0), i1)]}}[(0, 0)]
    1.9%   93.7%  0.281s  13.625s  1.41e-05s * 20000  2 GpuElemwise{sub,no_inplace}
    1.9%   95.5%  0.273s  13.898s  1.37e-05s * 20000  2 GpuElemwise{ScalarSigmoid{output_types_preference=transfer_type{0}}}[(0, 0)]
    1.9%   97.4%  0.273s  14.171s  1.37e-05s * 20000  2 GpuElemwise{Composite{[sub(i0, mul(i1, i2))]}}[(0, 0)]
    1.0%   98.4%  0.141s  14.313s  7.06e-06s * 20002  4 GpuDimShuffle{x}
    0.4%   98.8%  0.057s  14.370s  2.87e-06s * 20002  4 Elemwise{gt,no_inplace}
    0.3%   99.1%  0.048s  14.418s  1.21e-06s * 40002  6 Shape_i{0}
    0.3%   99.4%  0.045s  14.463s  2.25e-06s * 20000  2 InplaceDimShuffle{x}
    0.3%   99.7%  0.037s  14.500s  1.83e-06s * 20000  2 Elemwise{Cast{float32}}
    0.2%   99.8%  0.025s  14.525s  1.24e-06s * 20000  2 GpuDimShuffle{0}
    0.2%  100.0%  0.023s  14.548s  1.14e-06s * 20000  2 GpuDimShuffle{1,0}
   ... (remaining 1 Op account for   0.00%(0.00s) of the runtime)
(*) Op is running a c implementation

Apply-wise summary:
<% of local_time spent at this position> <cumulative %%> <apply time> <cumulative seconds> <time per call> [*] <nb_call> <Apply position> <Apply Op name>
   24.0%   24.0%  3.493s  3.493s 3.49e-04s    10000   1 GpuFromHost(x)
   23.9%   47.9%  3.479s  6.972s 3.48e-04s    10000   1 GpuFromHost(x)
    4.3%   52.3%  0.629s  7.602s 6.29e-05s  * 10000  24 GpuGemv{inplace}(w, TensorConstant{-0.00999999977648}, GpuDimShuffle{1,0}.0, GpuElemwise{Composite{[Composite{[Composite{[Composite{[mul(i0, add(i1, i2))]}(i0, neg(i1), true_div(i2, i3))]}(i0, mul(i1, i2, i3), i4, i5)]}(i0, i1, i2, exp(i3), i4, i5)]}}[(0, 0)].0, TensorConstant{0.999800026417})
    4.3%   56.6%  0.629s  8.231s 6.29e-05s  * 10000  24 GpuGemv{inplace}(w, TensorConstant{-0.00999999977648}, GpuDimShuffle{1,0}.0, GpuElemwise{Composite{[Composite{[Composite{[Composite{[mul(i0, add(i1, i2))]}(i0, neg(i1), true_div(i2, i3))]}(i0, mul(i1, i2, i3), i4, i5)]}(i0, i1, i2, exp(i3), i4, i5)]}}[(0, 0)].0, TensorConstant{0.999800026417})
    1.8%   58.4%  0.269s  8.499s 2.69e-05s  * 10000   9 GpuGemv{inplace}(GpuAlloc.0, TensorConstant{1.0}, GpuFromHost.0, w, TensorConstant{1.0})
    1.8%   60.3%  0.268s  8.767s 2.68e-05s  * 10000   9 GpuGemv{inplace}(GpuAlloc.0, TensorConstant{1.0}, GpuFromHost.0, w, TensorConstant{1.0})
    1.8%   62.1%  0.266s  9.033s 2.66e-05s    10000  18 HostFromGpu(GpuElemwise{Composite{[Composite{[Composite{[sub(mul(i0, i1), neg(i2))]}(i0, scalar_softplus(i1), mul(i2, i3))]}(i0, i1, i2, scalar_softplus(i3))]},no_inplace}.0)
    1.8%   63.9%  0.262s  9.296s 2.62e-05s    10000  18 HostFromGpu(GpuElemwise{Composite{[Composite{[Composite{[sub(mul(i0, i1), neg(i2))]}(i0, scalar_softplus(i1), mul(i2, i3))]}(i0, i1, i2, scalar_softplus(i3))]},no_inplace}.0)
    1.8%   65.7%  0.260s  9.555s 2.60e-05s    10000   3 GpuFromHost(y)
    1.8%   67.5%  0.258s  9.813s 2.58e-05s    10000   3 GpuFromHost(y)
    1.7%   69.2%  0.248s  10.061s 2.48e-05s    10000  20 HostFromGpu(GpuElemwise{ScalarSigmoid{output_types_preference=transfer_type{0}}}[(0, 0)].0)
    1.7%   70.9%  0.247s  10.309s 2.47e-05s    10000  20 HostFromGpu(GpuElemwise{ScalarSigmoid{output_types_preference=transfer_type{0}}}[(0, 0)].0)
    1.6%   72.5%  0.238s  10.547s 2.38e-05s    10000  12 GpuFromHost(Elemwise{Cast{float32}}.0)
    1.6%   74.1%  0.237s  10.785s 2.37e-05s    10000  12 GpuFromHost(Elemwise{Cast{float32}}.0)
    1.3%   75.4%  0.185s  10.969s 1.85e-05s  * 10000   6 GpuAlloc(CudaNdarrayConstant{[  1.58212732e-09]}, Shape_i{0}.0)
   ... (remaining 53 Apply instances account for 24.60%(3.58s) of the runtime)
(*) Op is running a c implementation

Some info useful for gpu:

    Spent 1.211s(8.324%) in cpu Op, 13.337s(91.676%) in gpu Op and 0.000s(0.000%) transfert Op

    Theano function input that are float64
    <fct name> <input name> <input type> <str input>

    List of apply that don't have float64 as input but have float64 in outputs
    (Useful to know if we forgot some cast when using floatX=float32 or gpu code)
    <Apply> <Apply position> <fct name> <inputs type> <outputs type>

Profile of Theano functions memory:
(This check only the output of each apply node. It don't check the temporary memory used by the op in the apply node.)
   We skipped 4 theano function(s). Each of them used less then 1024B(theano flags ProfileMode.min_memory_size) of total intermediate memory size

Here are tips to potentially make your code run faster
(if you think of new ones, suggest them on the mailing list).
Test them first, as they are not guaranteed to always provide a speedup.
  Sorry, no tip for today.



# 3. Conclusions


Facts:
Examine and compare 'Single Op-wise' summaries for CPU and GPU. GPU ops 'GpuFromHost' (and 'HostFromGpu') by themselves
consume a large amount of extra time, but by making as few as possible data transfers between GPU and CPU, you can minimize its overhead.
In addition, you probably need to increase the input data size (e.g. set N = 4000) to see the gain of the GPU.
Furthermore, notice that each of the GPU ops consumes more time than its CPU counterpart.
An additional experiment also confirms that adding an 'out' instance in the GPU version only brings about a minor
improvement in this situation.

Tentative conclusion:
The large number of external training steps (10000) generates disproportionate GPU overhead costs.

Tentative solution:
Include the training steps inside the definition of the Theano function.

Implement this solution and put it to test.


"""
