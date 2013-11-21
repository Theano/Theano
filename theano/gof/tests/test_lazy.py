from copy import deepcopy

import numpy

import theano
from theano.gof.op import PureOp
from theano.gof import Apply, generic, Container
from theano.gof.link import LocalLinker, map_storage, add_clear_storage
from theano import function, Mode
from theano.ifelse import ifelse
import theano.tensor as T


class IfElseIfElseIf(PureOp):

    def __init__(self, inplace=False):
        # check destroyhandler and others to ensure that a view_map with
        self.inplace = inplace
        #multiple inputs can work
        assert not self.inplace

    def make_node(self, c1, t1, c2, t2, c3, t3, f3):
        assert t1.type == f3.type
        assert t2.type == t3.type
        assert t3.type == f3.type
        return Apply(self, [c1, t1, c2, t2, c3, t3, f3], [t1.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):

        input_computed = [compute_map[v] for v in node.inputs]
        output_computed = [compute_map[v] for v in node.outputs]
        input_registers = [storage_map[v] for v in node.inputs]
        output_registers = [storage_map[v] for v in node.outputs]

        outtype = node.outputs[0].type

        def thunk():
            if not input_computed[0][0]:
                return [0]
            else:
                truthval = input_registers[0][0]
                if truthval:
                    if not input_computed[1][0]:
                        return [1]
                    else:
                        output_computed[0][0] = 1
                        output_registers[0][0] = outtype.filter(
                            deepcopy(input_registers[1][0]))
                        return []
                else:
                    if not input_computed[2][0]:
                        return [2]
                    else:
                        truthval = input_registers[2][0]
                        if truthval:
                            if not input_computed[3][0]:
                                return [3]
                            else:
                                output_computed[0][0] = 1
                                output_registers[0][0] = outtype.filter(
                                    deepcopy(input_registers[3][0]))
                                return []
                        else:
                            if not input_computed[4][0]:
                                return [4]
                            else:
                                truthval = input_registers[4][0]
                                if truthval:
                                    if not input_computed[5][0]:
                                        return [5]
                                    else:
                                        output_computed[0][0] = 1
                                        output_registers[0][0] = outtype.filter(
                                            deepcopy(input_registers[5][0]))
                                        return []
                                else:
                                    if not input_computed[6][0]:
                                        return [6]
                                    else:
                                        output_computed[0][0] = 1
                                        output_registers[0][0] = outtype.filter(
                                            deepcopy(input_registers[6][0]))
                                        return []

        thunk.lazy = True
        return thunk


class NotImplementedOp(PureOp):
    class E(Exception):
        pass

    def make_node(self, x):
        return Apply(self, [x], [x.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        def thunk():
            raise self.E()
        thunk.lazy = False
        return thunk


def test_ifelse():
    a = T.scalar()
    b = generic()
    c = generic()

    notimpl = NotImplementedOp()
    lazys = [True]
    # We need lazy to end up being True for this test.
    if theano.config.vm.lazy in [True, None]:
        lazys = [True, None]
    cloops = [True, False]
    if theano.config.cxx == "":
        cloops = [False]
    for cloop in cloops:
        for lazy in lazys:
            linker = theano.gof.vm.VM_Linker(use_cloop=cloop, lazy=lazy)
            f = function([a, b, c], ifelse(a, notimpl(b), c),
                         mode=Mode(linker=linker, optimizer='fast_run'))

            try:
                #print "case 1"
                f(1, 'a', 'b')
                assert False
            except NotImplementedOp.E:
                pass
            #print "... passed"

            #print "case 2"
            #print f(0, 'a', 'b')
            assert f(0, 'a', 'b') == 'b'
            #print "... passed"


def more_complex_test():
    notimpl = NotImplementedOp()
    ifelseifelseif = IfElseIfElseIf()

    x1 = T.scalar('x1')
    x2 = T.scalar('x2')
    c1 = T.scalar('c1')
    c2 = T.scalar('c2')
    t1 = ifelse(c1, x1, notimpl(x2))
    t1.name = 't1'
    t2 = t1 * 10
    t2.name = 't2'
    t3 = ifelse(c2, t2, x1 + t1)
    t3.name = 't3'
    t4 = ifelseifelseif(T.eq(x1, x2), x1, T.eq(x1, 5), x2, c2, t3, t3 + 0.5)
    t4.name = 't4'

    f = function([c1, c2, x1, x2], t4, mode=Mode(linker='vm',
                                                 optimizer='fast_run'))
    if theano.config.vm.lazy is False:
        try:
            f(1, 0, numpy.array(10, dtype=x1.dtype), 0)
            assert False
        except NotImplementedOp.E:
            pass
    else:
        print f(1, 0, numpy.array(10, dtype=x1.dtype), 0)
        assert f(1, 0, numpy.array(10, dtype=x1.dtype), 0) == 20.5
    print '... passed'

if __name__ == '__main__':
    more_complex_test()
