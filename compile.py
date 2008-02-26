import time, unittest
import numpy

import gof
import gof.lib
import cutils

import core
import opt
from copy import copy


def experimental_linker(env, target = None):
    order = env.toposort()
    
    for op in order:
        op.refresh()
        
    py_ops = set()
    thunks = []
    computed_results = []

    for op in order:
        try:
            factory = op.c_thunk_factory()
            for input in op.inputs:
                producer = input.owner
                if producer in py_ops:
                    result = lambda factory = factory: cutils.run_cthunk(factory())
                    break
            else:
                thunk = factory()
                result = lambda thunk = thunk: cutils.run_cthunk(thunk)
        except NotImplementedError:
            result = op._perform
            py_ops.add(op)
        thunks.append((result, op._perform_inplace))
        computed_results.extend(op.outputs)
    
    def ret():
        for thunk, fallback in thunks:
            try:
                thunk()
            except NotImplementedError:
                fallback()
        for r in computed_results:
            r.state = gof.result.Computed

    if not target:
        return ret
    else:
        raise NotImplementedError("Cannot write thunk representation to a file.")


class profile_linker:
    def __init__(self, env):
        self.order = env.toposort()
        self.thunks = [op._perform for op in self.order]
        self.n_calls = 0
        self.n_thunks = 0
        self.times = [0.0 for op in self.order]

    def print_for_dot(self):
        #TODO: popen2("dot -Tpng | display") and actually make the graph window pop up
         print "digraph unix { size = '6,6'; node [color = lightblue2; style = filled];"
         for op in self.order:
             for input in op.inputs:
                 if input.owner:
                     print input.owner.__class__.__name__ + str(abs(id(input.owner))), " -> ", op.__class__.__name__ + str(abs(id(op))), ";"
    
    def slow_call(self):
        """Run the program, timing each thunk.  """
        for i, thunk in enumerate(self.thunks):
            start_time = time.time()
            thunk()
            self.times[i] += time.time() - start_time
            self.n_thunks += 1
        self.n_calls += 1

    def fast_call(self):
        """Run the program, but only time the entire loop."""
        start_time = time.time()
        for th in self.thunks:
            th()
        self.n_thunks += len(self.thunks)
        self.n_calls += 1
        self.times[0] += time.time() - start_time

    __call__ = slow_call

    def dump(self, proportion=True):
        """Print statistics accumulated so far."""
        total_time = sum(self.times)
        print self.n_calls, 'calls took', total_time, 'seconds to evaluate',
        print self.n_thunks, 'thunks'

        if 0:
            print 'Proportion of CPU per op'
            for op, t in zip(self.order, self.times):
                s_op = str(op).split()[0][1:]
                print "  %-35s %4.5f"% (s_op, t/total_time)

        print 'Proportion of CPU per op class'
        dct = {}
        for op, t in zip(self.order, self.times):
            s_op = str(op).split()[0][1:]
            dct[s_op] = dct.get(s_op, 0.0) + t
        for t, s_op in reversed(sorted([(t,op) for op, t in dct.items()])):
            if proportion:
                print "  %-35s %4.5f"% (s_op, t/total_time)
            else:
                print "  %-35s %4.5f"% (s_op, t)




class prog(gof.Prog):
    def __init__(self, inputs, outputs, optimizer = opt.optimizer([]),
            linker = experimental_linker):
        """Compile a subgraph.

        N.B. This triggers computation of the subgraph leading to the outputs
        that is not fed by the inputs (the orphans).

        TODO: think about whether orphan computation should be in this function,
        or in self.__call__()
        """
        new_outputs = gof.mark_outputs_as_destroyed(outputs)
        gof.Prog.__init__(self,
                          inputs,
                          new_outputs,
                          optimizer,
                          linker,
                          [])
        self.outputs = outputs
        self.compute_orphans()

    def __call__(self, check_uncomputed = True):
        """Recompute the graph.

        If the inputs are uncomputed (and check_uncomputed is True) then an
        Exception is raised.
        """
        if check_uncomputed:
            for input in self.env.inputs:
                if input.data is None:
                    raise Exception("You must provide a value for input %s!" % input)
        return gof.Prog.__call__(self)

    def compute_orphans(self):
        for orphan in self.env.orphans():
            if orphan.data is None:
                if orphan.owner:
                    gof.lib.compute(orphan.owner)
                else:
                    raise Exception("Orphan %s is uncomputed but needed to calculate the function." % orphan)
            

def to_func(inputs, outputs):
#    print gof.Env(inputs, outputs).io_toposort()
##    p = prog([copy(input) for input in inputs], gof.graph.clone(inputs, outputs))
    p = prog(inputs, outputs)
    def f(*args):
        for input, value in zip(inputs, args):
            p[input] = value
        outputs = p()
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
    return f

def single(*outputs, **kwargs):
    return prog(gof.graph.inputs(outputs), outputs, **kwargs)

class _test_single_build_mode(unittest.TestCase):
    def setUp(self):
        core.build_mode()
        numpy.random.seed(44)
    def tearDown(self):
        core.pop_mode()

    def test_3(self):
        a = core.Numpy2(data=numpy.random.rand(2,2))
        b = core.Numpy2(data=numpy.random.rand(2,2))
        c = core.add(a,b)

        self.failUnless(c.data is None)
        self.failUnless(c.state is gof.result.Empty)
        p = single(c)
        self.failUnless(c.data is not None)
        self.failUnless(c.state is gof.result.Allocated)
        self.failUnless(not core._approx_eq(c, a.data + b.data))
        p()
        self.failUnless(c.state is gof.result.Computed)
        self.failUnless(core._approx_eq(c, a.data + b.data))

        new_a = numpy.random.rand(2,2)
        new_b = numpy.random.rand(2,2)
        a.data[:] = new_a
        b.data[:] = new_b
        p()
        self.failUnless(core._approx_eq(c, new_a + new_b))

    def test_get_element(self):
        core.build_eval_mode()
        a_data = numpy.random.rand(2,2)
        a = core.Numpy2(data=a_data)
        pos = core.input((0,0))
        a_i = core.get_slice(a, pos)
        p = single(a_i)
        #p()
        #print 'aaaa', a_i.owner.out, a_i.owner, a_i.data, pos.data

        #print 'pre p()'

        for i in 0,1:
            for j in 0,1:
                pos.data = (i,j)
                p()
                #print 'asdf', i,j,a_i.data
                #print a_i.owner.inputs[1].data
                #a_i.owner.inputs[1].data = [i,j]
                self.failUnless(a_data[i,j] == a_i.data)

        core.pop_mode()


if __name__ == '__main__':
    unittest.main()

