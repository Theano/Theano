#!/usr/bin/env python2.5
from __future__ import absolute_import
import numpy
import sys
import time

import theano
import theano.tensor as T
import theano.sandbox
import theano.sandbox.wraplinker
from theano.compile import module, Mode
from theano.sandbox.wraplinker import ProfileMode
from theano import gof, Op, Apply

from theano.tensor import blas, opt

# numpy: aa_numpy.py
# c : aa.cc


class _Dot22(Op):
    """Compute matrix-matrix product.
    This is a specialization of the more general Dot()
    """
    def make_node(self, x, y):
        assert x.type in T.float_matrix_types
        assert y.type == x.type
        bz = [x.type.broadcastable[0], y.type.broadcastable[1]]
        outputs = [T.tensor(x.type.dtype, bz)]
        return Apply(self, [x,y], outputs)

    def perform(self, node, (x, y), (z, )):
        try:
            z[0] = numpy.asarray(numpy.dot(x, y))
        except ValueError, e:
            # The error raised by numpy has no shape information, we mean to add that
            e.args = e.args + (x.shape, y.shape)
            raise
    def __str__(self):
        return "_dot22"
    def c_support_code(self):
        #return blas.cblas_header_text()
        mod_str = """
        #ifndef MOD
        #define MOD %
        #endif
        """
        return blas.blas_proto() + mod_str
    def c_headers(self):
        return ['<iostream>']
    def c_libraries(self):
        return blas.ldflags()
    def c_code(self, node, name, (_x, _y), (_z, ), sub):
        return """
        int unit = 0;

        int type_num = %(_x)s->descr->type_num;
        int type_size = %(_x)s->descr->elsize; // in bytes

        npy_intp* Nx = %(_x)s->dimensions;
        npy_intp* Ny = %(_y)s->dimensions;
        npy_intp* Nz = 0; //%(_z)s->dimensions;

        npy_intp* Sx = %(_x)s->strides;
        npy_intp* Sy = %(_y)s->strides;
        npy_intp* Sz = 0;//%(_z)s->strides;

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;

        if ((NULL == %(_z)s)
            || (%(_z)s->dimensions[0] != %(_x)s->dimensions[0])
            || (%(_z)s->dimensions[1] != %(_y)s->dimensions[1]))
        {
            if (NULL != %(_z)s) Py_XDECREF(%(_z)s);
            npy_intp dims[2];
            dims[0] = %(_x)s->dimensions[0];
            dims[1] = %(_y)s->dimensions[1];
            %(_z)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, type_num_%(_x)s);
            if(!%(_z)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc dot22 output");
                %(fail)s
            }
        }
        Nz = %(_z)s->dimensions;
        Sz = %(_z)s->strides;

        if (%(_x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(_y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(_z)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(z) != 2"); %(fail)s;}

        if ((%(_x)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_x)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); %(fail)s;}

        if ((%(_y)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_y)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); %(fail)s;}

        if ((%(_z)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_z)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); %(fail)s;}

        if ((%(_x)s->descr->type_num != %(_y)s->descr->type_num)
            ||(%(_x)s->descr->type_num != %(_z)s->descr->type_num))
        { PyErr_SetString(PyExc_NotImplementedError, "type(z), type(y), type(z) are not all the same"); %(fail)s; }

        if ((Nx[0] != Nz[0]) || (Nx[1] != Ny[0]) || (Ny[1] != Nz[1]))
        {
            PyErr_SetString(PyExc_ValueError, "Input dimensions do not agree");
            %(fail)s;
        }
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] MOD type_size) || (Sx[1] MOD type_size)
           || (Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] MOD type_size) || (Sy[1] MOD type_size)
           || (Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] MOD type_size) || (Sz[1] MOD type_size))
        {
            PyErr_SetString(PyExc_ValueError, "stride is not multiple of element size"); %(fail)s;
        }

        /*
        encode the stride structure of _x,_y,_z into a single integer
        */
        unit |= ((Sx[1] == type_size) ? 0x0 : (Sx[0] == type_size) ? 0x1 : 0x2) << 8;
        unit |= ((Sy[1] == type_size) ? 0x0 : (Sy[0] == type_size) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size) ? 0x0 : (Sz[0] == type_size) ? 0x1 : 0x2) << 0;

        /* create appropriate strides for malformed matrices that are row or column
         * vectors
         */
        sx_0 = (Nx[0] > 1) ? Sx[0]/type_size : Nx[1];
        sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : Nx[0];
        sy_0 = (Ny[0] > 1) ? Sy[0]/type_size : Ny[1];
        sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : Ny[0];
        sz_0 = (Nz[0] > 1) ? Sz[0]/type_size : Nz[1];
        sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : Nz[0];

        switch (type_num)
        {
            case PyArray_FLOAT:
            {
                float a = 1.0;
                float b = 0.0;
                float* x = (float*)PyArray_DATA(%(_x)s);
                float* y = (float*)PyArray_DATA(%(_y)s);
                float* z = (float*)PyArray_DATA(%(_z)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
                switch(unit)
                {
                    case 0x000: sgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: sgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: sgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: sgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: sgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: sgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: sgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: sgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
                };
                #undef REAL
            }
            break;
            case PyArray_DOUBLE:
            {
                double a = 1.0;
                double b = 0.0;
                double* x = (double*)PyArray_DATA(%(_x)s);
                double* y = (double*)PyArray_DATA(%(_y)s);
                double* z = (double*)PyArray_DATA(%(_z)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
                switch(unit)
                {
                    case 0x000: dgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: dgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: dgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: dgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: dgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: dgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: dgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: dgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
                };
                #undef REAL
            }
            break;
        }

        """ % dict(locals(), **sub)
_dot22 = _Dot22()

@gof.local_optimizer([T.dot])
def local_dot_to_dot22(node):
    if node.op == T.dot:
        return [_dot22(*node.inputs)]
    else:
        return False
T.opt.register_specialize(local_dot_to_dot22)

@gof.local_optimizer([T.sub])
def local_sub_to_gemm(node):
    if node.op == T.sub:
        subleft, subright = node.inputs
        if subright.owner and (subright.owner.op == _dot22):
            dotleft, dotright = subright.owner.inputs
            return [T.gemm(subleft, -1.0, dotleft, dotright, 1.0)]
        if subright.owner and (subright.owner.op == T.mul):
            mulleft, mulright = subright.owner.inputs
            #TODO: we actually want to get any scalar here, not necessrily a constant
            mulleft_const = opt.local_mul_canonizer.get_constant(mulleft)
            if mulleft_const is not None:
                assert mulleft_const.size() == 1
                mulleft_const = mulleft_const.flatten()[0]
                #subleft - (mulleft_const * ?)
                if mulright.owner and (mulright.owner.op == T.add):
                    #subleft - (mulleft_const * (? + ?))
                    addleft, addright = mulright.owner.inputs
                    if addright.owner and addright.owner.op == T.DimShuffle([False,False], [1,0]):
                        #subleft - (mulleft_const * (? + ?.T))
                        raise NotImplementedError()
                    if addright.owner and addright.owner.op == T.DimShuffle([False,False], [1,0], inplace=True):
                        #subleft - (mulleft_const * (? + ?.T))
                        transposed = addright.owner.inputs[0]
                        if transposed.owner and transposed.owner.op == _dot22:
                            x, y = transposed.owner.inputs
                            #subleft - (mulleft_const * (addleft + dot(x, y).T))
                            if addleft.owner and addleft.owner.op == _dot22:
                                u, v = addleft.owner.inputs
                                #subleft - (mulleft_const * (dot(u,v) + dot(x, y).T))
                                return [T.gemm(
                                    T.gemm(subleft, -mulleft_const, y.T, x.T, 1.0),
                                    -mulleft_const, u, v, 1.0)]

            if mulright.owner and (mulright.owner.op == _dot22):
                dotleft, dotright = mulright.owner.inputs
                #TODO: we actually want to get any scalar here, not necessrily a constant
                mulleft_const = opt.local_mul_canonizer.get_constant(mulleft)
                if mulleft_const:
                    return [T.gemm(subleft, -mulleft_const, dotleft, dotright, 1.0)]
            if mulleft.owner and (mulleft.owner.op == _dot22):
                dotleft, dotright = mulleft.owner.inputs
                #TODO: we actually want to get any scalar here, not necessrily a constant
                mulright_const = opt.local_mul_canonizer.get_constant(mulright)
                if mulright_const:
                    return [T.gemm(subleft, -mulright_const, dotleft, dotright, 1.0)]
    return False
T.opt.register_specialize(local_sub_to_gemm)


if 0:
    class Opt(object):
        merge = theano.gof.MergeOptimizer()
        gemm_opt_1 = theano.gof.TopoOptimizer(theano.tensor_opt.gemm_pattern_1)

        gemm_opt_2 = theano.gof.TopoOptimizer( # d -= a * (dot()+transpose(dot))
                theano.gof.PatternSub(
                    (
                        T.sub_inplace,
                        'd',
                        (
                            T.mul,
                            dict(pattern = (T.DimShuffle((), ['x', 'x'], inplace = True), 'a'),
                                allow_multiple_clients = True),
                            (
                                T.add,
                                (T.dot, 'b', 'c'),
                                (T.transpose_inplace, (T.dot, 'f', 'g'))
                            )
                        )
                    ),
                    (
                        T.gemm, 
                        (
                            T.gemm,
                            'd', 
                            (T.neg, 'a'),
                            (T.transpose_inplace, 'g'),
                            (T.transpose_inplace, 'f'),
                            T.constant(1.0)
                        ),
                        (T.neg, 'a'), 
                        'b', 
                        'c', 
                        T.constant(1.0)
                    ),
                    allow_multiple_clients = False))

        sqr = []
        sqr.append( theano.gof.TopoOptimizer(
                theano.gof.PatternSub(
                    (T.mul,'x', 'x'),
                    (T.sqr, 'x'), allow_multiple_clients=True)))
        sqr.append(theano.gof.TopoOptimizer(
            theano.gof.PatternSub(
                (T.pow, 'x', (T.DimShuffle((), ['x', 'x'], inplace=True), T.constant(2))),
                (T.sqr, 'x'), allow_multiple_clients=True)))

        ident_opt_list = []
        ident_opt_list.append(  # remove explicit copies
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.tensor_copy, 'x'),
                        'x',
                        allow_multiple_clients=True)))
        ident_opt_list.append( # remove double-transpose
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.transpose_inplace, (T.transpose_inplace, 'x')),
                        'x',
                        allow_multiple_clients=True)))

        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.sqr, (T.sqrt,'x')),
                        'x',
                        allow_multiple_clients=True)))
        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.sqrt, (T.sqr,'x')),
                        'x',
                        allow_multiple_clients=True)))
        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.mul, 'x', (T.div,'y', 'x')),
                        'y',
                        allow_multiple_clients=True)))

        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.mul, (T.div,'y', 'x'), 'x'),
                        'y',
                        allow_multiple_clients=True)))

        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.div, (T.mul,'y', 'x'), 'x'),
                        'y',
                        allow_multiple_clients=True)))

        ident_opt_list.append(
                theano.gof.TopoOptimizer(
                    theano.gof.PatternSub(
                        (T.div, (T.mul,'y', 'x'), 'y'),
                        'x',
                        allow_multiple_clients=True)))

        def __call__(self, env):
            self.merge(env)
            #eliminate identities
            if 0:
                print 'SKIPPING optimizations'
            else:

                for opt in self.ident_opt_list:
                    opt(env)

                for opt in self.sqr:
                    opt(env)

                self.gemm_opt_1(env)
                self.gemm_opt_2(env)

                self.merge(env)

def print_graph_linker(print_prog=True):
    if 1:
        imap = {None:'-'}
        def blah(i, node, thunk):
            imap[node] = str(i)
            if print_prog:# and node.op.__class__ is T.DimShuffle:
                if False and  node.op == T.DimShuffle((), ['x', 'x'], inplace = True):
                    print node.op == T.DimShuffle((), ['x', 'x'], inplace = True),
                    print node.inputs[0], type(node.inputs[0]), 
                    print node.inputs[0].equals(T.constant(2)), 
                outputs = node.outputs
                inputs = theano.gof.graph.inputs(outputs)
                print 'node ', i, node,
                print ':'.join([imap[inp.owner] for inp in node.inputs])
                #print theano.sandbox.pprint.pp.process_graph(inputs, outputs)
        return theano.sandbox.wraplinker.WrapLinkerMany(
                [theano.gof.OpWiseCLinker()],
                [theano.sandbox.wraplinker.run_all
                    ,blah
                    #,theano.sandbox.wraplinker.numpy_notall_isfinite
                    ])
    else:
        return theano.gof.OpWiseCLinker()


class M(module.Module):
    def __init__(self):
        super(M, self).__init__()

        x = T.matrix('x') # input, target
        self.w = module.Member(T.matrix('w')) # weights
        self.a = module.Member(T.vector('a')) # hid bias
        self.b = module.Member(T.vector('b')) # output bias

        self.hid = T.tanh(T.dot(x, self.w) + self.a)
        hid = self.hid

        self.out = T.tanh(T.dot(hid, self.w.T) + self.b)
        out = self.out

        self.err = 0.5 * T.sum((out - x)**2)
        err = self.err

        params = [self.w, self.a, self.b]

        gparams = T.grad(err, params)

        updates = [(p, p - 0.01 * gp) for p, gp in zip(params, gparams)]

        self.step = module.Method([x], err, updates=dict(updates))

mod = M()
mode = 'FAST_RUN'
#mode = ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
mode = Mode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker(nice_errors=True))
mode = Mode(optimizer='fast_run', linker='c')
print mod.pretty(mode=mode)
m = mod.make(mode=mode)

neg, nout, nhid, niter = [int(a) for a in sys.argv[1:]]
rng = numpy.random.RandomState(342)
m.w = rng.rand(nout, nhid)
m.a = rng.randn(nhid) * 0.0
m.b = rng.randn(nout) * 0.0

x = (rng.rand(neg, nout)-0.5) * 1.5

t = time.time()
for i in xrange(niter):
    err = m.step(x)
print 'time: ',time.time() - t, 'err: ', err
try:
    mode.print_summary()
    pass
except:
    pass


