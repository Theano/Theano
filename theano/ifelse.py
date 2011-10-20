"""
IfElse introduces lazy evaluation in Theano (coupled with the CVM/VM
linkers). It resembles the if clause of any programming languages, that
has a `then` and `else` branch, and executes either one or the other
according to the condition provided.

This op contrast the already existent `swtich` op, that will evaluate both
branches of the clause and afterwards pick (according to the condition)
which value to report. Note also that `switch` is an elemwise operations (so
it picks each entry of a matrix according to the condition) while `ifelse`
is a global operation with a scalar condition.
"""

__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
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

_logger = logging.getLogger('theano.ifelse')


class IfElse(PureOp):
    """
    Op that provides conditional graph evaluation if used with the CVM/VM
    linkers. Note that there exist a helpful function `ifelse` that should
    be used to instantiate the op!

    According to a scalar condition `condition` the op evaluates and then
    returns all the tensors provided on the `then` branch, otherwise it
    evaluates and returns the tensors provided on the `else` branch. The op
    supports multiple tensors on each branch, conditioned that the same
    number of tensors are on the `then` as on the `else` and there is a one
    to one correspondance between them (shape and dtype wise).

    The `then` branch is defined as the first N tensors (after the
    condition), while the `else` branch is defined as the last N tensors.

    Example usage:

        ``rval = ifelse(condition, rval_if_true1, .., rval_if_trueN,
                        rval_if_false1, rval_if_false2, .., rval_if_falseN)``

    :note:
        Other Linkers then CVM and VM are INCOMPATIBLE with this Op, and
        will ingnore its lazy characteristic, computing both the True and
        False branch before picking one.

    """
    def __init__(self, n_outs, as_view=False, gpu=False, name=None):
        if as_view:
            # check destroyhandler and others to ensure that a view_map with
            # multiple inputs can work
            view_map = {}
            for idx in xrange(n_outs):
                view_map[idx] = [idx + 1]
            self.view_map = view_map
        self.as_view = as_view
        self.gpu = gpu
        self.n_outs = n_outs
        self.name = name

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
        rval = (hash(type(self)) ^
                hash(self.as_view) ^
                hash(self.gpu) ^
                hash(self.n_outs))
        return rval

    def __str__(self):
        name = 'if{%s' % str(self.name)
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
                return [None] * self.n_outs

        new_fs_inputs = []
        for fs_shape in fs_shapes:
            if isinstance(fs_shape, (list, tuple)):
                new_fs_inputs += list(fs_shape)
            else:
                # It can be None for generic objects
                return [None] * self.n_outs

        assert len(new_ts_inputs) == len(new_fs_inputs)
        if len(new_ts_inputs + new_fs_inputs) > 0:

            new_ifelse = IfElse(
                n_outs=len(new_ts_inputs),
                as_view=False,
                gpu=False,
                name='shape_' + str(self.name))
            new_outs = new_ifelse.make_node(node.inputs[0],
                            *(new_ts_inputs + new_fs_inputs)).outputs
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

        for t, f in zip(ts, fs):
            if t.type != f.type:
                raise TypeError(('IfElse requires same types for true and '
                                'false return values'), t, f, t.type, f.type)
        if c.ndim > 0:
            raise TypeError(('Condition given to the op has to be a scalar '
                             'with 0 standing for False, anything else '
                             'for True'))
        return Apply(self, [c] + list(args), [t.type() for t in ts])

    def R_op(self, inputs, eval_points):
        return self.make_node(inputs[0], *eval_points[1:]).outputs

    def grad(self, ins, grads):
        ts = ins[1:][:self.n_outs]
        fs = ins[1:][self.n_outs:]
        if self.name is not None:
            nw_name_t = self.name + '_grad_t'
            nw_name_f = self.name + '_grad_f'
        else:
            nw_name_t = None
            nw_name_f = None
        if_true_op = IfElse(n_outs=self.n_outs,
                            as_view=self.as_view,
                            gpu=self.gpu,
                            name=nw_name_t)

        if_false_op = IfElse(n_outs=self.n_outs,
                             as_view=self.as_view,
                             gpu=self.gpu,
                             name=nw_name_f)

        if_true = ([ins[0]] + grads + [theano.tensor.zeros_like(t)
                                     for t in ts])
        if_false = ([ins[0]] + [theano.tensor.zeros_like(f)
                                for f in fs] + grads)
        return ([None] +
                if_true_op.make_node(*if_true).outputs +
                if_false_op.make_node(*if_false).outputs)

    def make_thunk(self, node, storage_map, compute_map, no_recycling):

        outtypes = [out.type for out in node.outputs]
        cond = node.inputs[0]
        ts = node.inputs[1:][:self.n_outs]
        fs = node.inputs[1:][self.n_outs:]
        outputs = node.outputs

        def thunk():
            if not compute_map[cond][0]:
                return [0]
            else:
                truthval = storage_map[cond][0]
                if truthval != 0:
                    ls = [idx + 1 for idx in xrange(self.n_outs)
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
                    ls = [1 + idx + self.n_outs for idx in xrange(self.n_outs)
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
        thunk.inputs = [storage_map[v] for v in node.inputs]
        thunk.outputs = [storage_map[v] for v in node.outputs]
        return thunk


def ifelse(condition, then_branch, else_branch, name=None):
    """
    This function corresponds to an if statement, returning (and evaluating)
    inputs in the ``then_branch`` if ``condition`` evaluates to True or
    inputs in the ``else_branch`` if ``condition`` evalutates to False.

    :type condition: scalar like
    :param condition:
        ``condition`` should be a tensor scalar representing the condition.
        If it evaluates to 0 it corresponds to False, anything else stands
        for True.

    :type then_branch: list of theano expressions/ theano expressions
    :param then_branch:
        A single theano variable or a list of theano variables that the
        function should return as the output if ``condition`` evaluates to
        true. The number of variables should match those in the
        ``else_branch``, and there should be a one to one correspondance
        (type wise) with the tensors provided in the else branch

    :type else_branch: list of theano expressions/ theano expressions
    :param else_branch:
        A single theano variable or a list of theano variables that the
        function should return as the output if ``condition`` evaluates to
        false. The number of variables should match those in the then branch,
        and there should be a one to one correspondace (type wise) with the
        tensors provided in the then branch.

    :return:
        A list of theano variables or a single variable (depending on the
        nature of the ``then_branch`` and ``else_branch``). More exactly if
        ``then_branch`` and ``else_branch`` is a tensor, then
        the return variable will be just a single variable, otherwise a
        list. The value returns correspond either to the values in the
        ``then_branch`` or in the ``else_branch`` depending on the value of
        ``cond``.
    """
    if type(then_branch) is not type(else_branch):
        raise ValueError(('The two branches should be identical. '
                          'This error could be raised if for example '
                          ' you provided a one element list on the then '
                          ' branch but a tensor on the else branch'))

    rval_type = None
    if type(then_branch) is list:
        rval_type = list
    elif type(then_branch) is tuple:
        rval_type = tuple

    if type(then_branch) not in (list, tuple):
        then_branch = [then_branch]
    if type(else_branch) not in (list, tuple):
        else_branch = [else_branch]

    if len(then_branch) != len(else_branch):
        raise ValueError(('The number of values on the `then` branch'
                          ' should have the same number of variables as '
                          'the `else` branch : (variables on `then` '
                          '%d' % len(then_branch) + ', variables on `else` '
                          '%d' % len(else_branch) + ')'))

    new_ifelse = IfElse(n_outs=len(then_branch),
                        as_view=False,
                        gpu=False,
                        name=name)

    ins = [cond] + list(then_branch) + list(else_branch)
    rval = new_ifelse.make_node(*ins).outputs

    if rval_type is None:
        return rval[0]
    elif rval_type is list:
        return list(rval)
    else:
        return tuple(rval)


@gof.local_optimizer([None])
def cond_make_inplace(node):
    op = node.op
    if isinstance(op, IfElse) and not op.as_view:
        return IfElse(n_outs=op.n_outs,
                      as_view=True,
                      gpu=op.gpu,
                      name=op.name).make_node(*node.inputs).outputs
    return False

optdb.register('cond_make_inplace', opt.in2out(cond_make_inplace,
    ignore_newtrees=True), 95, 'fast_run', 'inplace')

ifelse_equilibrium = gof.EquilibriumDB()
ifelse_seqopt = gof.SequenceDB()
ifelse_equilibrium.register('seq_ifelse', ifelse_seqopt, 'fast_run',
                            'ifelse')
optdb.register('ifelse_equilibriumOpt', ifelse_equilibrium, .5, 'fast_run',
               'ifelse')


@gof.local_optimizer([None])
def cond_merge_ifs_true(node):
    op = node.op
    if not isinstance(op, IfElse):
        return False
    t_ins = node.inputs[1:][:op.n_outs]

    replace = {}
    for idx, tval in enumerate(t_ins):
        if (tval.owner and isinstance(tval.owner.op, IfElse) and
            tval.owner.inputs[0] == node.inputs[0]):
            ins_op = tval.owner.op
            ins_t = tval.owner.inputs[1:][:ins_op.n_outs]
            replace[idx + 1] = ins_t[tval.owner.outputs.index(tval)]

    if len(replace.items()) == 0:
        return False

    old_ins = list(node.inputs)
    for pos, var in replace.items():
        old_ins[pos] = var
    return op.make_node(*old_ins).outputs


@gof.local_optimizer([None])
def cond_merge_ifs_false(node):
    op = node.op
    if not isinstance(op, IfElse):
        return False
    f_ins = node.inputs[1:][op.n_outs:]

    replace = {}
    for idx, fval in enumerate(f_ins):
        if (fval.owner and isinstance(fval.owner.op, IfElse) and
            fval.owner.inputs[0] == node.inputs[0]):
            ins_op = fval.owner.op
            ins_t = fval.owner.inputs[1:][ins_op.n_outs:]
            replace[idx + 1 + op.n_outs] = \
                    ins_t[fval.owner.outputs.index(fval)]

    if len(replace.items()) == 0:
        return False

    old_ins = list(node.inputs)
    for pos, var in replace.items():
        old_ins[pos] = var
    return op.make_node(*old_ins).outputs


class CondMerge(gof.Optimizer):
    """ Graph Optimizer that merges different cond ops """
    def add_requirements(self, env):
        env.extend(gof.toolbox.ReplaceValidate())

    def apply(self, env):
        nodelist = list(env.toposort())
        cond_nodes = filter(lambda s: isinstance(s.op, IfElse), nodelist)
        if len(cond_nodes) < 2:
            return False
        merging_node = cond_nodes[0]
        for proposal in cond_nodes[1:]:
            if (proposal.inputs[0] == merging_node.inputs[0] and
                not find_up(proposal, merging_node)):
                # Create a list of replacements for proposal
                mn_ts = merging_node.inputs[1:][:merging_node.op.n_outs]
                mn_fs = merging_node.inputs[1:][merging_node.op.n_outs:]
                pl_ts = proposal.inputs[1:][:proposal.op.n_outs]
                pl_fs = proposal.inputs[1:][proposal.op.n_outs:]
                new_ins = ([merging_node.inputs[0]] +
                           mn_ts + pl_ts + mn_fs + pl_fs)
                mn_name = '?'
                if merging_node.op.name:
                    mn_name = merging_node.op.name
                pl_name = '?'
                mn_n_ts = len(mn_ts)
                mn_n_fs = len(mn_fs)
                if proposal.op.name:
                    pl_name = proposal.op.name
                new_ifelse = IfElse(
                    n_outs=len(mn_ts + pl_ts),
                    as_view=False,
                    gpu=False,
                    name=mn_name + '&' + pl_name)
                print 'here'
                new_outs = new_ifelse.make_node(*new_ins).outputs
                new_outs = [clone(x) for x in new_outs]
                old_outs = []
                if type(merging_node.outputs) not in (list, tuple):
                    old_outs += [merging_node.outputs]
                else:
                    old_outs += merging_node.outputs
                if type(proposal.outputs) not in (list, tuple):
                    old_outs += [proposal.outputs]
                else:
                    old_outs += proposal.outputs
                pairs = zip(old_outs, new_outs)
                env.replace_all_validate(pairs, reason='cond_merge')


@gof.local_optimizer([None])
def cond_remove_identical(node):
    op = node.op

    if not isinstance(op, IfElse):
        return False
    ts = node.inputs[1:][:op.n_outs]
    fs = node.inputs[1:][op.n_outs:]

    # sync outs
    out_map = {}
    for idx in xrange(len(node.outputs)):
        if idx not in out_map:
            for jdx in xrange(idx + 1, len(node.outputs)):
                if (ts[idx] == ts[jdx] and
                    fs[idx] == fs[jdx] and
                    jdx not in out_map):
                    out_map[jdx] = idx

    if len(out_map.keys()) == 0:
        return False

    nw_ts = []
    nw_fs = []
    inv_map = {}
    pos = 0
    for idx in xrange(len(node.outputs)):
        if idx not in out_map:
            inv_map[idx] = pos
            pos = pos + 1
            nw_ts.append(ts[idx])
            nw_fs.append(fs[idx])

    new_ifelse = IfElse(n_outs=len(nw_ts),
                        as_view=op.as_view,
                        gpu=op.gpu,
                        name=op.name)

    new_ins = [node.inputs[0]] + nw_ts + nw_fs
    new_outs = new_ifelse.make_node(*new_ins).outputs

    rval = []
    for idx in xrange(len(node.outputs)):
        if idx in out_map.keys():
            rval += [new_outs[inv_map[out_map[idx]]]]
        else:
            rval += [new_outs[inv_map[idx]]]

    return rval

acceptable_ops = (theano.tensor.basic.Dot,
                  theano.tensor.basic.Reshape,
                  theano.tensor.basic.Shape,
                  theano.tensor.basic.SpecifyShape,
                  theano.tensor.basic.MaxAndArgmax,
                  theano.tensor.basic.Subtensor,
                  theano.tensor.basic.IncSubtensor,
                  theano.tensor.basic.Rebroadcast,
                  theano.tensor.basic.Alloc,
                  theano.tensor.elemwise.Elemwise,
                  theano.tensor.elemwise.DimShuffle)


@gof.local_optimizer([None])
def cond_lift_single_if(main_node):
    if not (isinstance(main_node.op, acceptable_ops)):
        return False
    all_inp_nodes = set()
    for inp in main_node.inputs:
        all_inp_nodes.add(inp.owner)
    ifnodes = [x for x in list(all_inp_nodes)
                    if x and isinstance(x.op, IfElse)]
    # if we have multiple ifs as inputs .. it all becomes quite complicated
    # :)
    if len(ifnodes) != 1:
        return False
    node = ifnodes[0]
    op = node.op

    ts = node.inputs[1:][:op.n_outs]
    fs = node.inputs[1:][op.n_outs:]

    outs = main_node.outputs
    mop = main_node.op
    true_ins = []
    false_ins = []

    for x in main_node.inputs:
        if x in node.outputs:
            idx = node.outputs.index(x)
            true_ins.append(ts[idx])
            false_ins.append(fs[idx])
        else:
            true_ins.append(x)
            false_ins.append(x)
    true_eval = mop.make_node(*true_ins).outputs
    false_eval = mop.make_node(*false_ins).outputs
    #true_eval  = clone(outs, replace = dict(zip(node.outputs, ts)))
    #false_eval = clone(outs, replace = dict(zip(node.outputs, fs)))

    nw_outs = ifelse(node.inputs[0], true_eval, false_eval)
    if type(nw_outs) not in (tuple, list):
        nw_outs = [nw_outs]
    return nw_outs


@gof.local_optimizer([None])
def cond_merge_random_op(main_node):
    if isinstance(main_node.op, IfElse):
        return False

    all_inp_nodes = set()
    for inp in main_node.inputs:
        all_inp_nodes.add(inp.owner)
    cond_nodes = [x for x in list(all_inp_nodes)
                        if x and isinstance(x.op, IfElse)]

    if len(cond_nodes) < 2:
        return False

    merging_node = cond_nodes[0]
    for proposal in cond_nodes[1:]:
        if (proposal.inputs[0] == merging_node.inputs[0] and
            not find_up(proposal, merging_node) and
            not find_up(merging_node, proposal)):
            # Create a list of replacements for proposal
            mn_ts = merging_node.inputs[1:][:merging_node.op.n_outs]
            mn_fs = merging_node.inputs[1:][merging_node.op.n_outs:]
            pl_ts = proposal.inputs[1:][:proposal.op.n_outs]
            pl_fs = proposal.inputs[1:][proposal.op.n_outs:]
            new_ins = ([merging_node.inputs[0]] +
                       mn_ts + pl_ts + mn_fs + pl_fs)
            mn_name = '?'
            if merging_node.op.name:
                mn_name = merging_node.op.name
            pl_name = '?'
            mn_n_ts = len(mn_ts)
            mn_n_fs = len(mn_fs)
            if proposal.op.name:
                pl_name = proposal.op.name
            new_ifelse = IfElse(
                n_outs=len(mn_ts + pl_ts),
                as_view=False,
                gpu=False,
                name=mn_name + '&' + pl_name)
            new_outs = new_ifelse.make_node(*new_ins).outputs
            old_outs = []
            if type(merging_node.outputs) not in (list, tuple):
                old_outs += [merging_node.outputs]
            else:
                old_outs += merging_node.outputs
            if type(proposal.outputs) not in (list, tuple):
                old_outs += [proposal.outputs]
            else:
                old_outs += proposal.outputs
            pairs = zip(old_outs, new_outs)
            main_outs = clone(main_node.outputs, replace=pairs)
            return main_outs


pushout_equilibrium = gof.EquilibriumDB()

pushout_equilibrium.register("ifelse_lift",
                             opt.in2out(cond_lift_single_if,
                                        ignore_newtrees=True),
                             'fast_run', 'ifelse')

pushout_equilibrium.register("ifelse_merge_ifs",
                             opt.in2out(cond_merge_random_op,
                                        ignore_newtrees=True),
                             'fast_run', 'ifelse')


pushout_equilibrium.register("ifelse_merge_nodes",
                             gof.MergeOptimizer(skip_const_merge=False),
                             'fast_run', 'ifelse')

pushout_equilibrium.register("ifelse_remove_identical_inside",
                       opt.in2out(cond_remove_identical,
                                  ignore_newtrees=True),
                             'fast_run', 'ifelse')

pushout_equilibrium.register('ifelse_sameCondTrue_inside',
                       opt.in2out(cond_merge_ifs_true,
                                  ignore_newtrees=True),
                       'fast_run', 'ifelse')

pushout_equilibrium.register('ifelse_sameCondFalse_inside',
                       opt.in2out(cond_merge_ifs_false,
                                  ignore_newtrees=True),
                       'fast_run', 'ifelse')


ifelse_seqopt.register('ifelse_condPushOut_equilibrium',
                       pushout_equilibrium,
                       1, 'fast_run', 'ifelse')

ifelse_seqopt.register('merge_nodes_1',
                       gof.MergeOptimizer(skip_const_merge=False),
                       2, 'fast_run', 'ifelse')


ifelse_seqopt.register('ifelse_sameCondTrue',
                       opt.in2out(cond_merge_ifs_true,
                                  ignore_newtrees=True),
                       3, 'fast_run', 'ifelse')


ifelse_seqopt.register('ifelse_sameCondFalse',
                       opt.in2out(cond_merge_ifs_false,
                                  ignore_newtrees=True),
                       4, 'fast_run', 'ifelse')


ifelse_seqopt.register('ifelse_removeIdenetical',
                       opt.in2out(cond_remove_identical,
                                  ignore_newtrees=True),
                       7, 'fast_run', 'ifelse')
