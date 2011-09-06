"""
IfElse is an Op that works with the LazyLinker to support conditional graph evaluation.

:TODO: Add text to library documentation describing the IfElse Op.
"""

__docformat__ = 'restructedtext en'
__authors__ = ( "Razvan Pascanu "
                "James Bergstra "
                "Dumitru Erhan ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

from copy import deepcopy
import logging

from theano.gof import PureOp, Apply, generic, Container

import theano.tensor
import gof

from compile import optdb
from tensor import opt
from scan_module.scan_utils import find_up
from scan_module.scan_utils import clone

_logger = logging.getLogger('theano.lazycond')




class IfElse(PureOp):
    """
    Op that works with CVM/VM to support conditional graph evaluation.

    Example usage:

        ``rval = ifelse(tf, rval_if_true1, rval_if_true2, .., rval_if_trueN,
                        rval_if_false1, rval_if_false2, .., rval_if_falseN)``

    :note:
        Other Linkers then CVM and VM are INCOMPATIBLE with this Op, and
        will ingnore its lazy characteristic, computing both the True and
        False branch before picking one.

    """
    def __init__(self, n_outs, as_view=False, gpu = False, name = None):
        if as_view:
            # check destroyhandler and others to ensure that a view_map with
            # multiple inputs can work
            view_map = {}
            for idx in xrange(n_outs):
                view_map[idx] = [idx+1]
            self.view_map = view_map
            #raise NotImplementedError('Cond must copy for now')
        self.as_view = as_view
        self.gpu    = gpu
        self.n_outs = n_outs
        self.name   = name


    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        if not self.as_view == other.as_view:
            return False
        if not self.gpu == other.gpu:
            return False
        if not self.n_outs == other.n_outs:
            return False
        return True

    def __hash__(self):

        rval = ( hash(type(self)) ^
                hash(self.as_view) ^
                hash(self.gpu) ^
                hash(self.n_outs))

        return rval

    def __str__(self):
        name ='if{%s'%str(self.name)
        if self.as_view:
            name += ',inplace'
        if self.gpu:
            name += ',gpu'
        name += '}'
        return name


    def infer_shape(self, node, inputs_shapes):
        # By construction, corresponding then/else pairs have the same number
        # of dimensions
        ts_shapes = inputs_shapes[1:][:self.n_outs]
        fs_shapes = inputs_shapes[1:][self.n_outs:]
        new_ts_inputs = []
        for ts_shape in ts_shapes:
            if isinstance(ts_shape, (list, tuple)):
                new_ts_inputs += list(ts_shape)
            else:
                # It can be None for generic objects
                return [None]*self.n_outs

        new_fs_inputs = []
        for fs_shape in fs_shapes:
            if isinstance(fs_shape, (list, tuple)):
                new_fs_inputs += list(fs_shape)
            else:
                # It can be None for generic objects
                return [None]*self.n_outs

        assert len(new_ts_inputs) == len(new_fs_inputs)
        if len(new_ts_inputs + new_fs_inputs) > 0:

            new_ifelse = IfElse(
                n_outs = len(new_ts_inputs),
                as_view = False,
                gpu = False,
                name='shape_'+str(self.name))
            new_outs = new_ifelse.make_node(node.inputs[0],
                                            *(new_ts_inputs+new_fs_inputs)).outputs
        else:
            new_outs = []

        # generate pairs of shapes
        out_shapes = []

        idx = 0
        for out in node.outputs:
            current_shape = []
            for k in xrange(out.ndim):
                current_shape += [new_outs[idx]]
                idx += 1
            out_shapes += [tuple(current_shape)]
        return out_shapes




    def R_op(self, inputs, eval_points):
        return self.make_node(inputs[0],*eval_points[1:]).outputs


    def make_node(self, c, *args):
        if not self.gpu:
            # When gpu is true, we are given only cuda ndarrays, and we want
            # to keep them be cuda ndarrays
            c = theano.tensor.as_tensor_variable(c)
            nw_args = []
            for x in args:
                if isinstance(x, theano.Variable):
                    nw_args.append(x)
                else:
                    nw_args.append(theano.tensor.as_tensor_variable(x))
            args = nw_args
        ts = args[:self.n_outs]
        fs = args[self.n_outs:]

        for t,f in zip(ts, fs):
            if t.type != f.type:
                raise TypeError(('IfElse requires same types for true and '
                                'false return values'), t, f, t.type, f.type)
        if c.ndim >0:
            raise TypeError(('Condition given to the op has to be a scalar '
                            'with 0 standing for False, anything else for True'))
        return Apply(self, [c]+list(args), [t.type() for t in ts])


    def grad(self, ins, grads):
        ts = ins[1:][:self.n_outs]
        fs = ins[1:][self.n_outs:]
        if self.name is not None:
            nw_name_t = self.name + '_grad_t'
            nw_name_f = self.name + '_grad_f'
        else:
            nw_name_t = None
            nw_name_f = None
        if_true_op = IfElse(n_outs = self.n_outs,
                            as_view = self.as_view,
                            gpu  = self.gpu,
                            name = nw_name_t)

        if_false_op = IfElse(n_outs = self.n_outs,
                             as_view = self.as_view,
                             gpu = self.gpu,
                             name = nw_name_f)

        if_true = ([ins[0]]+ grads+ [theano.tensor.zeros_like(t)
                                     for t in ts])
        if_false = ([ins[0]] + [theano.tensor.zeros_like(f)
                                for f in fs] + grads)
        return ([None]+
                if_true_op.make_node(*if_true).outputs +
                if_false_op.make_node(*if_false).outputs )




    def make_thunk(self, node, storage_map, compute_map, no_recycling):

        outtypes = [ out.type for out in node.outputs]
        cond     = node.inputs[0]
        ts = node.inputs[1:][:self.n_outs]
        fs = node.inputs[1:][self.n_outs:]
        outputs  = node.outputs


        def thunk():
            if not compute_map[cond][0]:
                return [0]
            else:
                truthval = storage_map[cond][0]
                if truthval != 0:
                    ls = [idx+1 for idx in xrange(self.n_outs)
                          if not compute_map[ts[idx]][0]]
                    if len(ls) > 0:
                        return ls
                    else:
                        for out, outtype, t in zip(outputs,
                                                   outtypes,
                                                   ts):
                            compute_map[out][0] = 1
                            if self.as_view:
                                oval = outtype.filter(storage_map[t][0])
                            else:
                                oval = outtype.filter(
                                    deepcopy(storage_map[t][0]))
                            storage_map[out][0] = oval
                        return []
                else:
                    ls = [1+idx+self.n_outs for idx in xrange(self.n_outs)
                          if not compute_map[fs[idx]][0]]
                    if len(ls) > 0:
                        return ls
                    else:
                        for out, outtype, f in zip(outputs,
                                                   outtypes,
                                                   fs):
                            compute_map[out][0] = 1
                            # can't view both outputs unless destroyhandler
                            # improves
                            oval = outtype.filter(
                                deepcopy(storage_map[f][0]))
                            storage_map[out][0] = oval
                        return []

        thunk.lazy = True
        thunk.inputs  = [ storage_map[v] for v in node.inputs]
        thunk.outputs = [ storage_map[v] for v in node.outputs]
        return thunk


def ifelse( cond, true_branch, false_branch, name = None):
    """
    This function corresponds to a if statement, returning inputs in the
    ``true_branch`` if ``cond`` evaluates to True or inputs in the
    ``false_branch`` if ``cond`` evalutates to False.

    :param cond:
        ``cond`` should be a tensor scalar representing the condition. If it
        evaluates to 0 it corresponds to False, anything else stands for
        True.

    :param true_branch:
        A single theano variable or a list of theano variables that the
        function should return as the output if ``cond`` evaluates to true.
        The number of variables should match those in the false_branch, and
        the types (of each) should also correspond to those in the false
        branch.

    :param false_branch:
        A single theano variable or a list of theano variables that the
        function should return as the output if ``cond`` evaluates to false.
        The number of variables should match those in the true branch, and
        the types (of each) should also match those in the true branch.

    :return:
        A list of theano variables or a single variable ( depending on the
        nature of the ``true_branch`` and ``false_branch``). More exactly if
        ``true_branch`` and ``false_branch`` contain a single element, then
        the return variable will be just a single variable, otherwise a
        list. The value returns correspond either to the values in the
        ``true_branch`` or in the ``false_branch`` depending on the value of
        ``cond``.
    """
    if type(true_branch) not in (list, tuple):
        true_branch = [true_branch]
    if type(false_branch) not in (list, tuple):
        false_branch = [false_branch]

    assert len(true_branch) == len(false_branch)
    new_ifelse = IfElse(n_outs = len(true_branch),
                        as_view=False,
                        gpu = False,
                        name = name)

    ins = [cond] + list(true_branch) + list(false_branch)
    rval = new_ifelse.make_node(*ins).outputs
    if type(rval) in (list,tuple) and len(rval) == 1:
        return rval[0]
    else:
        return rval



@gof.local_optimizer([None])
def cond_make_inplace(node):
    op = node.op
    if isinstance(op, IfElse) and not op.as_view :
        return IfElse(n_outs  = op.n_outs,
                      as_view = True,
                      gpu   = op.gpu,
                      name  = op.name ).make_node(*node.inputs).outputs
    return False

optdb.register('cond_make_inplace', opt.in2out(cond_make_inplace,
    ignore_newtrees=True), 95, 'fast_run', 'inplace')

ifelse_equilibrium = gof.EquilibriumDB()
ifelse_seqopt = gof.SequenceDB()
ifelse_equilibrium.register('seq_ifelse', ifelse_seqopt, 'fast_run',
                            'ifelse')
optdb.register('ifelse_equilibriumOpt', ifelse_equilibrium, .5, 'fast_run', 'ifelse')

