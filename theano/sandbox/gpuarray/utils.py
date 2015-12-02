import copy

from .basic_ops import GpuFromHost, GpuToGpu, GpuAlloc, GpuAllocEmpty, GpuEye
from .type import GpuArrayType

context_ops = (GpuFromHost, GpuToGpu, GpuAlloc, GpuAllocEmpty, GpuEye)


def _clone_type(t, context_map):
    if isinstance(t, GpuArrayType) and t.context_name in context_map:
        t_copy = copy.copy(t)
        t_copy.context_name = context_map[t.context_name]
        return t_copy
    return None


def _clone_v(v, context_map):
    new_type = _clone_type(v.type, context_map)
    if new_type is None:
        return v.clone()
    else:
        return v.clone_with_new_type(new_type)


def _clone_apply(n, new_inputs, context_map):
    if n.op in context_ops and n.op.context_name in context_map:
        new_op = copy.copy(n.op)
        new_op.context_name = context_map[n.op.context_name]
        new_n = new_op.make_node(*new_inputs)
        new_n.tag = copy.copy(n.tag).__update__(new_n.tag)
        return new_n
    return n.clone_with_new_inputs(new_inputs, strict=False)


def clone_swap_dev(f, context_map, name=None, profile=None):
    """
    This function clones a compiled function, swapping GPU context
    names.

    This can be useful if you want to run the same model on more than
    one GPU as it will cut down on recompile time.

    By default shared varaiables will not be cloned, but they can be.
    However only shared variables in one the `old` context will be
    clone (and they will be to the mapped `new` context).

    .. warning::

        The devices that map to the names must be of the same class
        otherwise the cloned function might not run properly.

    Parameters
    ----------
    f : Function
        A Theano function.
    context_map : dict
        A dictionary mapping old context names to new ones.
    name : string
        Name for the new function (defaults to `f.name + 'copy'`)
    profile : object
        See the profile argument of :meth:`theano.function`.
    """
    return f.copy(clone_var=lambda v, cmap=context_map: _clone_v(v, cmap),
                  clone_apply=lambda n, inps, cmap=context_map:
                      _clone_apply(n, inps, cmap),
                  name=name, profile=profile)
