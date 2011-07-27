"""
IfElse is an Op that works with the LazyLinker to support conditional graph evaluation.

:TODO: Add text to library documentation describing the IfElse Op.
"""
from copy import deepcopy
import logging

from theano.gof import PureOp, Apply, generic, Container

import theano.tensor
import gof

from compile import optdb
from tensor import opt

_logger = logging.getLogger('theano.lazycond')

@gof.local_optimizer([None])
def ifelse_make_inplace(node):
    op = node.op
    if isinstance(op, IfElse) and not op.as_view :
        _logger.debug('ifelse_make_inplace applied')
        return IfElse(as_view = True,
                    gpu = op.gpu, name=op.name).make_node(*node.inputs).outputs
    return False

optdb.register('ifelse_make_inplace', opt.in2out(ifelse_make_inplace,
    ignore_newtrees=True), 95, 'fast_run', 'inplace')


class IfElse(PureOp):
    """
    Op that works with LazyLinker to support conditional graph evaluation.
    
    Example usage:

        ``rval = ifelse(tf, rval_if_true, rval_if_false)``

    :type tf: symbolic tensor
    :param tf: boolean variable representing a condition
    :type rval_if_true: symbolic tensor
    :param rval_if_false: symbolic variable to compute if tf is True
    :type rval_if_false: symbolic tensor
    :param rval_if_false: symbolic variable to compute if tf is False
    :return: tensor corresponding to rval_if_true if tf is True or
             rval_if_false if tf is False

    While the switch function computes both values (rval_if_true and rval_if_false),
    the ifelse op only computes one e.g rval_if_true is computed if tf is True.
    
    :note:
        Other Linkers (ALL other linkers right now) are INCOMPATIBLE with this
        Op, they will produce functions that FAIL TO EXECUTE.

    """

    def __init__(self, as_view=False, gpu = False, name = None):
        if as_view:
            # check destroyhandler and others to ensure that a view_map with
            # multiple inputs can work
            view_map = {}
            view_map[0] = [1]
            self.view_map = view_map
            #raise NotImplementedError('IfElse must copy for now')
        else:
            self.view_map = {}
        self.as_view=as_view
        self.gpu = gpu
        self.name = name

    def __eq__(self, other):
        return (type(self)==type(other) and
                self.as_view == other.as_view and
                # view_map included in as_view
                #self.view_map == other.view_map and
                self.gpu == other.gpu and
                self.name == other.name)

    def __hash__(self, other):
        return (hash(type(self)) ^
                # view_map included in as_view
                # and dict are not hashable
                #hash(self.view_map) ^
                hash(self.as_view) ^
                hash(self.gpu) ^
                hash(self.name))

    def make_node(self, c, t, f):
        if t.type != f.type:
            raise TypeError(
                    'IfElse requires same types for true and false args',
                    (t.type, f.type))
        return Apply(self, [c,t,f], [t.type()])


    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        outtype = node.outputs[0].type
        c,t,f = node.inputs
        output = node.outputs[0]
        def thunk():
            if not compute_map[c][0]:
                return [0]
            else:
                truthval = storage_map[c][0]
                if truthval:
                    if not compute_map[t][0]:
                        return [1]
                    else:
                        compute_map[output][0]=1
                        if self.as_view:
                            oval = outtype.filter(storage_map[t][0])
                        else:
                            oval = outtype.filter(
                                    deepcopy(storage_map[t][0]))
                        storage_map[output][0] = oval
                        return []
                else:
                    if not compute_map[f][0]:
                        return [2]
                    else:
                        # can't view both outputs unless destroyhandler
                        # improves
                        compute_map[output][0]=1
                        oval = outtype.filter(
                                deepcopy(storage_map[f][0]))
                        storage_map[output][0]=oval
                        return []
        thunk.lazy = True
        thunk.inputs  = [storage_map[v] for v in node.inputs]
        thunk.outputs = [storage_map[v] for v in node.outputs]

        return thunk

ifelse = IfElse()
