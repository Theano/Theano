from __future__ import absolute_import, print_function, division

import numpy as np
from six import integer_types
from six.moves import StringIO, xrange

from theano import tensor, gof, Op
from theano.gof import ParamsType
from theano.gradient import grad_not_implemented
import theano.tensor as T
from theano.tensor.subtensor import IncSubtensor, Subtensor, get_idx_list
from theano.tensor import AllocDiag
from theano.scalar import bool as bool_t, int32 as int_t, uint32 as size_t

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .type import GpuArrayType, gpu_context_type
from .basic_ops import (as_gpuarray_variable, HideC, GpuKernelBase, Kernel, gpuarray_helper_inc_dir,
                        infer_context_name, gpu_contiguous)

iadd_reg = {}


def get_iadd(a, b):
    key = (a.type.dtype, b.type.dtype, a.type.context)
    if key not in iadd_reg:
        a_arg = pygpu.elemwise.arg('a', a.type.dtype, read=True, write=True)
        b_arg = pygpu.elemwise.arg('b', b.type.dtype, read=True)
        res = pygpu.elemwise.GpuElemwise(a.type.context, "a = a + b", [a_arg, b_arg], convert_f16=True)
        iadd_reg[key] = res
    return iadd_reg[key]


class GpuSubtensor(HideC, Subtensor):
    """
    Subtensor on the GPU.
    """
    _f16_ok = True

    def make_node(self, x, *inputs):
        ctx_name = infer_context_name(x)
        rval = tensor.Subtensor.make_node(self, x, *inputs)
        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
                             broadcastable=rval.outputs[0].type.broadcastable,
                             context_name=ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        return gof.Apply(self, [x] + rval.inputs[1:], [otype()])

    def perform(self, node, inputs, out_):
        out, = out_
        x = inputs[0]

        cdata = get_idx_list(inputs, self.idx_list)
        if len(cdata) == 1:
            cdata = cdata[0]

        out[0] = x.__getitem__(cdata)

    def c_support_code(self):
        return """
        static int fix_indices(ssize_t *start, ssize_t *stop, ssize_t *step,
                               int start_n, int stop_n, int step_n,
                               size_t len) {
            if (step_n) *step = 1;
            if (*step == 0) {
                PyErr_SetString(PyExc_ValueError, "slice step cannot be zero");
                return -1;
            }
            if (start_n) *start = (*step < 0) ? len-1 : 0;
            else {
                if (*start < 0) *start += len;
                if (*start < 0) *start = (*step < 0) ? -1 : 0;
                if (*start > -1 && *start >= len) {
                    *start = (*step < 0) ? len-1 : len;
                }
            }

            if (stop_n) *stop = (*step < 0) ? -1 : len;
            else {
                if (*stop < 0) *stop += len;
                if (*stop < 0) *stop = (*step < 0) ? -1 : 0;
                if (*stop > -1 && *stop >= len) {
                    *stop = (*step < 0) ? len-1 : len;
                }
            }
            if (*stop < *start && *step > 0)
                *stop = *start;
            return 0;
        }
        """

    def c_code(self, node, name, inputs, outputs, sub):
        inp_ndim = node.inputs[0].ndim
        inp = inputs[0]
        indices = inputs[1:]

        # pad out the index list to the same dimension as the input
        idx_list = self.idx_list + \
            ((slice(None),) * (inp_ndim - len(self.idx_list)))

        # This case fails when we use pygpu_index(), so here is some
        # special code
        if len(idx_list) == 0:
            return """
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_copy(%(inp)s, GA_ANY_ORDER);
        if (!%(out)s) {
            // Exception already set
            %(fail)s
        }
""" % dict(out=outputs[0], inp=inp, fail=sub['fail'])

        sio = StringIO()
        print("""
        ssize_t starts[%(sz)s];
        ssize_t stops[%(sz)s];
        ssize_t steps[%(sz)s];
        ssize_t cur;
        int err;

        if (%(inp)s->ga.nd != %(sz)s) {
            PyErr_SetString(PyExc_IndexError, "invalid index");
            %(fail)s
        }
        """ % dict(sz=len(idx_list), inp=inp, fail=sub['fail']), file=sio)

        def fix_idx(idx):
            if idx is None:
                return "0", 1
            elif isinstance(idx, (np.integer, integer_types)):
                return str(idx), 0
            elif isinstance(idx, gof.Type):
                return indices.pop(0), 0
            else:
                assert 0, idx

        for i, idx in enumerate(idx_list):
            if isinstance(idx, slice):
                start, start_n = fix_idx(idx.start)
                stop, stop_n = fix_idx(idx.stop)
                step, step_n = fix_idx(idx.step)
                print("""
                starts[%(i)s] = %(start)s;
                stops[%(i)s] = %(stop)s;
                steps[%(i)s] = %(step)s;
                if (fix_indices(&starts[%(i)s], &stops[%(i)s], &steps[%(i)s],
                                %(start_n)s, %(stop_n)s, %(step_n)s,
                                %(inp)s->ga.dimensions[%(i)s]) == -1) {
                    %(fail)s
                }
                """ % dict(i=i, start=start, stop=stop, step=step,
                           start_n=start_n, stop_n=stop_n, step_n=step_n,
                           fail=sub['fail'], inp=inp), file=sio)
            else:
                if isinstance(idx, gof.Type):
                    start = indices.pop(0)
                elif isinstance(idx, (np.integer, integer_types)):
                    start = idx
                else:
                    assert 0, idx
                print("""
                cur = %(start)s;
                if (cur < 0)
                    cur += %(inp)s->ga.dimensions[%(i)s];
                starts[%(i)s] = cur;
                steps[%(i)s] = 0;
                """ % dict(i=i, start=start, fail=sub['fail'], inp=inp), file=sio)

        print("""
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_index(%(inp)s, starts, stops, steps);
        if (!%(out)s) { %(fail)s }
""" % dict(name=name, fail=sub['fail'], inp=inp, out=outputs[0]), file=sio)

        return sio.getvalue()

    def c_code_cache_version(self):
        return (8,)


class GpuIncSubtensor(IncSubtensor):
    """
    Implement IncSubtensor on the gpu.

    Notes
    -----
    The optimization to make this inplace is in tensor/opt.
    The same optimization handles IncSubtensor and GpuIncSubtensor.
    This Op has c_code too; it inherits tensor.IncSubtensor's c_code.
    The helper methods like :meth:`do_type_checking`,
    :meth:`copy_of_x`, etc. specialize the c_code for this Op.

    """
    _f16_ok = True
    params_type = gpu_context_type

    def make_node(self, x, y, *inputs):
        ctx_name = infer_context_name(x, y)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        rval = tensor.IncSubtensor.make_node(self, x, y, *inputs)
        ret = gof.Apply(self, [x, y] + rval.inputs[2:], [x.type()])
        return ret

    def get_params(self, node):
        return node.outputs[0].type.context

    def perform(self, node, inputs, out_, ctx):
        out, = out_
        x, y = inputs[:2]
        indices = list(reversed(inputs[2:]))

        def convert(entry):
            if isinstance(entry, gof.Type):
                rval = indices.pop()
                return rval
            elif isinstance(entry, slice):
                return slice(convert(entry.start),
                             convert(entry.stop),
                             convert(entry.step))
            else:
                return entry

        cdata = tuple(map(convert, self.idx_list))
        if len(cdata) == 1:
            cdata = cdata[0]
        if not self.inplace:
            x = x.copy()
        sub_x = x.__getitem__(cdata)
        if sub_x.shape:
            # we've sliced out an N-D tensor with N > 0
            if not self.set_instead_of_inc:
                # sub_x += y
                iadd = get_iadd(node.inputs[0], node.inputs[1])
                iadd(sub_x, y)
            else:
                # sub_x[...] = y
                x.__setitem__(cdata, y)
        else:
            # scalar case
            if not self.set_instead_of_inc:
                # x.__setitem__(cdata, sub_x + y)
                tmp = pygpu.elemwise.elemwise2(sub_x, '+', y, sub_x,
                                               broadcast=False)
                x.__setitem__(cdata, tmp)
            else:
                x.__setitem__(cdata, y)
        out[0] = x

    def do_type_checking(self, node):
        """
        Should raise NotImplementedError if c_code does not support
        the types involved in this node.

        """

        if not isinstance(node.inputs[0].type, GpuArrayType):
            raise NotImplementedError()

    def copy_of_x(self, x):
        """

        Parameters
        ----------
        x
            A string giving the name of a C variable pointing to an array.

        Returns
        -------
        str
            C code expression to make a copy of x.

        Notes
        -----
        Base class uses `PyArrayObject *`, subclasses may override for
        different types of arrays.

        """
        return """pygpu_copy(%(x)s, GA_ANY_ORDER)""" % locals()

    def decl_view(self):
        return "PyGpuArrayObject* zview = NULL;"

    def make_view_array(self, x, view_ndim):
        """
        //TODO

        Parameters
        ----------
        x
            A string identifying an array to be viewed.
        view_ndim
            A string specifying the number of dimensions to have in the view.
            This doesn't need to actually set up the view with the
            right indexing; we'll do that manually later.

        """
        ret = """
        size_t dims[%(view_ndim)s];
        for(int i=0; i<%(view_ndim)s; i++)
            dims[i] = xview_dims[i];

        zview = pygpu_fromgpudata(%(x)s->ga.data,
                                  %(x)s->ga.offset + xview_offset,
                                  %(x)s->ga.typecode,
                                  %(view_ndim)s,
                                  dims,
                                  xview_strides,
                                  %(x)s->context,
                                  1,
                                  (PyObject *)%(x)s,
                                  (PyObject *)&PyGpuArrayType);
        """ % locals()
        return ret

    def get_helper_c_code_args(self):
        """
        Return a dictionary of arguments to use with helper_c_code.

        """
        return {'c_prefix': 'PyGpuArray',
                'strides_mul': 1
                }

    def copy_into(self, view, source):
        """

        Parameters
        ----------
        view : string
            C code expression for an array.
        source : string
            C code expression for an array.

        Returns
        -------
        str
            C code expression to copy source into view, and 0 on success.

        """
        return """sub_setarray(&%(view)s->ga, &%(source)s->ga)""" % locals()

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/error.h>', '<gpuarray/array.h>',
                '<gpuarray/elemwise.h>']

    def c_support_code(self):
        return """
int sub_setarray(GpuArray *dst, GpuArray *src) {
  int err;
  err = GpuArray_setarray(dst, src);
  if (err != GA_NO_ERROR)
    PyErr_SetString(PyExc_RuntimeError, GpuArray_error(src, err));
  return err;
}
"""

    def c_support_code_struct(self, node, nodename):
        return "\nGpuElemwise *iadd;\n"

    def c_init_code_struct(self, node, name, sub):
        return """
        gpuelemwise_arg args[2] = {{0}};
        args[0].name = "a";
        args[0].typecode = %(type1)s;
        args[0].flags = GE_READ|GE_WRITE;
        args[1].name = "b";
        args[1].typecode = %(type2)s;
        args[1].flags = GE_READ;
        iadd = GpuElemwise_new(%(ctx)s->ctx, "", "a += b",
                               2, args, %(nd)s, GE_CONVERT_F16);
        if (iadd == NULL) {
          PyErr_SetString(PyExc_RuntimeError, "Could not intialize inplace add support");
          %(fail)s
        }
        """ % dict(ctx=sub['params'], fail=sub['fail'],
                   type1=node.inputs[0].type.typecode,
                   type2=node.inputs[1].type.typecode,
                   nd=node.inputs[1].ndim)

    def add_to_zview(self, nodename, x, fail):
        return """
        {
          void *args[2];
          args[0] = &zview->ga;
          args[1] = &%(x)s->ga;
          if (GpuElemwise_call(iadd, args, GE_BROADCAST | GE_PADSHAPE) != GA_NO_ERROR) {
            PyErr_SetString(PyExc_RuntimeError, "Error doing inplace add");
            Py_DECREF(zview);
            %(fail)s
          }
        }
        """ % locals()

    def c_code_cache_version(self):
        parent_version = super(GpuIncSubtensor, self).c_code_cache_version()
        if not parent_version:
            return
        return parent_version + (10,)


class GpuAdvancedSubtensor1(HideC, tensor.AdvancedSubtensor1):
    """
    AdvancedSubrensor1 on the GPU.
    """
    _f16_ok = True

    def make_node(self, x, ilist):
        ctx_name = infer_context_name(x, ilist)
        x_ = as_gpuarray_variable(x, ctx_name)

        ilist__ = tensor.as_tensor_variable(ilist)
        if ilist__.type.dtype not in tensor.integer_dtypes:
            raise TypeError('index must be integers')
        if ilist__.type.dtype != 'int64':
            ilist__ = tensor.cast(ilist__, 'int64')

        ilist_ = gpu_contiguous(as_gpuarray_variable(ilist__, ctx_name))

        if ilist_.type.dtype != 'int64':
            raise TypeError('index must be int64')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be a vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')

        bcast = ilist_.broadcastable + x_.broadcastable[1:]
        return gof.Apply(self, [x_, ilist_],
                         [GpuArrayType(dtype=x.dtype,
                                       context_name=ctx_name,
                                       broadcastable=bcast)()])

    def perform(self, node, inp, out_):
        raise NotImplementedError()

    def c_support_code(self):
        return """
int take1_match_dims(GpuArray *a, GpuArray *v) {
  if (a->nd != v->nd) return 0;
  for (unsigned int i = 1; i < v->nd; i++) {
    if (a->dimensions[i] != v->dimensions[i]) return 0;
  }
  return 1;
}
"""

    def c_code(self, node, name, inputs, outputs, sub):
        return """
int err;
if (%(out)s == NULL || !GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga) ||
    %(out)s->ga.dimensions[0] != %(idx)s->ga.dimensions[0] ||
    !take1_match_dims(&%(out)s->ga, &%(v)s->ga)) {
  size_t tmp;
  Py_XDECREF(%(out)s);

  /* This is a dirty hack to avoid an extra alloc */
  tmp = %(v)s->ga.dimensions[0];
  %(v)s->ga.dimensions[0] = %(idx)s->ga.dimensions[0];
  %(out)s = pygpu_empty(%(v)s->ga.nd, %(v)s->ga.dimensions, %(v)s->ga.typecode,
                        GA_C_ORDER, %(v)s->context, Py_None);
  if (%(out)s == NULL) {
    %(fail)s;
  }
  %(v)s->ga.dimensions[0] = tmp; // Don't remove this line
}

err = GpuArray_take1(&%(out)s->ga, &%(v)s->ga, &%(idx)s->ga, 1);
if (err != GA_NO_ERROR) {
  if (err == GA_VALUE_ERROR) {
    PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
  } else {
    PyErr_SetString(PyExc_RuntimeError, GpuArray_error(&%(v)s->ga, err));
  }
  %(fail)s
}
""" % dict(out=outputs[0], v=inputs[0], idx=inputs[1], fail=sub['fail'])

    def c_code_cache_version(self):
        return (1,)


def check_and_convert_boolean_masks(input, idx_list):
    """
    This function checks if the boolean mask arrays in the index have
    the right shape and converts them to index arrays by calling nonzero.
    For each boolean mask, we check if the mask has the
    same shape as the input. This is enforced in NumPy 0.13.0 and
    newer, but not by earlier versions. If the size is not the same,
    this method raises an IndexError.
    """
    dim_seen = 0
    out_idx_list = []
    for index in idx_list:
        if index is np.newaxis:
            # skip, does not count as an input dimension
            out_idx_list.append(index)
        elif isinstance(index, np.ndarray) and index.dtype == 'bool':
            for i in xrange(index.ndim):
                if index.shape[i] != input.shape[dim_seen + i]:
                    raise IndexError('boolean index did not match indexed array '
                                     'along dimension %d; dimension is %d but '
                                     'corresponding boolean dimension is %d' %
                                     (dim_seen + i, input.shape[dim_seen + i],
                                      index.shape[i]))
            dim_seen += index.ndim
            out_idx_list += index.nonzero()
        else:
            dim_seen += 1
            out_idx_list.append(index)
    return out_idx_list


class BaseGpuAdvancedSubtensor(object):
    def perform(self, node, inputs, out_):
        out, = out_
        x = inputs[0]
        idx = inputs[1:]

        # convert boolean masks to index arrays
        idx = check_and_convert_boolean_masks(x, idx)

        # detect and transpose array indices
        nidx = []
        nshp = list(x.shape)
        for k, i in enumerate(idx):
            if i is None:
                nidx.append(slice(None))
                nshp.insert(k, 1)
            else:
                nidx.append(i)

        x = x.reshape(nshp)

        transp = list(range(x.ndim))
        # number of array-indexed dimensions
        p = 0
        # ap represents the axis in the resulting array where the
        # dimensions indexed by arrays and ints will be inserted.
        # For instance, if all such dimensions are grouped together,
        # it corresponds to the index of the first such dimension in the
        # initial array.  If these dimensions are split (with slices
        # between), then the resulting dimensions will be moved to the
        # beginning, and ap will be 0.
        # If no such dimension has been encountered, ap is None.
        ap = None
        # Indicates whether we have already encountered an index (array
        # or number), and then a slice.
        slice_after_idx = False
        for k, i in enumerate(list(nidx)):
            if (isinstance(i, np.ndarray) and i.ndim != 0):
                transp.remove(k)
                transp.insert(p, k)
                i = nidx.pop(k)
                nidx.insert(p, i)
                p += 1
                if ap is None:
                    # first non-slice index
                    ap = k
                elif slice_after_idx:
                    # We already encountered at least an array or int, and then
                    # a slice. Array-indexed axes are not grouped,
                    # moving to the beginning
                    ap = 0
            else:
                try:
                    i.__index__()
                    if ap is None:
                        ap = k
                    # indices do not break the contiguity of
                    # array-indexed axes
                except Exception:
                    # If we already encountered an array/int index, it
                    # means future ones will not be grouped.
                    if ap is not None:
                        slice_after_idx = True

        x = x.transpose(*transp)

        idx_ = ([slice(None)] * p + nidx[p:])
        x = x.__getitem__(idx_)

        if p == 0:
            assert ap is None
            # The only indexing was through slices and indices.
            # This can happen with symbolic slices for instance.
            # Since no view_map is set, we need to copy the returned value
            out[0] = x.copy()
            return

        # At this point, we should have encountered at least one array
        assert ap is not None

        # flatten the array-indexed dimensions
        shape = ((np.prod(x.shape[0: p]),) +
                 x.shape[p:])
        input_flat = x.reshape(shape)

        # build the strides
        strides = [1]
        for i in range(p - 1, 0, -1):
            stride = x.shape[i] * strides[0]
            strides.insert(0, stride)

        # build the indices and use it
        take_idx = sum((i * s for i, s in zip(nidx, strides)))
        out_flat = input_flat.take1(pygpu.asarray(take_idx.flatten(),
                                                  context=x.context))

        # finish up
        out_flat_shp = take_idx.shape + x.shape[p:]
        o = out_flat.reshape(out_flat_shp)

        if ap != 0:
            # Put the resulting indexing at the place that NumPy
            # decided was the right one.
            ntransp = list(range(take_idx.ndim, o.ndim))
            ntransp[ap:ap] = list(range(take_idx.ndim))
            o = o.transpose(*ntransp)

        out[0] = o


class GpuAdvancedSubtensor(HideC, BaseGpuAdvancedSubtensor, tensor.AdvancedSubtensor):
    """
    AdvancedSubtensor on the GPU.
    """
    def make_node(self, x, *inputs):
        ctx_name = infer_context_name(x)
        # This method relies on AdvancedSubtensor.make_node to
        # call tensor.subtensor.check_and_reject_bool(inputs),
        # which raises an IndexError if there are any boolean indices.
        rval = tensor.AdvancedSubtensor.make_node(self, x, *inputs)
        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
                             broadcastable=rval.outputs[0].type.broadcastable,
                             context_name=ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        return gof.Apply(self, [x] + rval.inputs[1:], [otype()])


class GpuAdvancedBooleanSubtensor(HideC, BaseGpuAdvancedSubtensor, tensor.AdvancedBooleanSubtensor):
    """
    AdvancedBooleanSubtensor on the GPU.
    """
    def make_node(self, x, *inputs):
        ctx_name = infer_context_name(x)
        rval = tensor.AdvancedBooleanSubtensor.make_node(self, x, *inputs)
        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
                             broadcastable=rval.outputs[0].type.broadcastable,
                             context_name=ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        return gof.Apply(self, [x] + rval.inputs[1:], [otype()])


class BaseGpuAdvancedIncSubtensor(object):
    def perform(self, node, inp, out_):
        out, = out_
        x = inp[0]
        y = inp[1]
        idx = inp[2:]
        x = x.copy()
        # Get a handle to the GpuElemwise object that will be called.
        # It is not necessary to have the right number of dimensions,
        # so we just pass symbolic x and y.
        iadd = get_iadd(node.inputs[0], node.inputs[1])

        # convert all indices to np.array
        for i in range(len(idx)):
            if isinstance(idx[i], gpuarray.GpuArray):
                idx[i] = np.asarray(idx[i])

        # convert boolean masks to index arrays
        idx = check_and_convert_boolean_masks(x, idx)

        # Insert axes for None indexing
        nidx = []
        nshp = list(x.shape)
        for k, i in enumerate(idx):
            if i is None:
                nidx.append(slice(None))
                nshp.insert(k, 1)
            else:
                nidx.append(i)

        x_ = x.reshape(nshp)

        # Bring array indices to front
        transp = []
        nidx_ = []
        p = 0
        for k, i in enumerate(list(nidx)):
            if isinstance(i, np.ndarray) and i.ndim != 0:
                transp.append(k)
                nidx_.append(i)
                p += 1
        for k, i in enumerate(list(nidx)):
            if not (isinstance(i, np.ndarray) and i.ndim != 0):
                transp.append(k)
                nidx_.append(i)
        transp = transp + list(range(len(transp), x_.ndim))
        rtransp = [i for i, _ in sorted(enumerate(transp), key=lambda x:x[1])]
        nidx = nidx_

        # transp: order to shuffle axes of x so that single dimension
        #         subarrays are extracted first
        # p: number of axes with array indexing
        x_ = x_.transpose(*transp)
        idx_ = ([slice(None)] * p + nidx[p:])
        # flatten the array-indexed dimensions
        x_flat = x_.reshape((np.prod(x_.shape[0: p]),) + x_.shape[p:])
        # process y so that last axes are the same
        if y.shape != (1,):
            y_shape_reverse = []
            for x_s, y_s in zip(x_flat.shape[::-1], y.shape[::-1]):
                if x_s == y_s or y_s == 1:
                    y_shape_reverse.append(y_s)
                else:
                    break
            if np.prod(y_shape_reverse) < np.prod(y.shape):
                if len(y_shape_reverse) > 0:
                    y_shape_reverse.append(
                        int(np.prod(y.shape[0:-len(y_shape_reverse)])))
                else:
                    y_shape_reverse.append(int(np.prod(y.shape)))

            y_shape = y_shape_reverse[::-1]
            y_flat = y.reshape(y_shape)
        else:
            y_flat = y[0]

        # build the strides
        strides = [1]
        for i in range(p - 1, 0, -1):
            stride = x_.shape[i] * strides[0]
            strides.insert(0, stride)

        # build the indices and use it
        index = idx_[p:] + [slice(None)] * (len(x_flat.shape) - len(idx_[p:]) - 1)
        take_idx = sum(i * s for i, s in zip(nidx, strides))
        if index == []:
            for j, i in enumerate(take_idx.flatten()):
                if y_flat.shape == ():
                    val = y_flat
                else:
                    val = y_flat[j]

                iadd(x_flat[i], val, broadcast=True)
        else:
            if (x_flat.shape[-len(y_flat.shape):] == y_flat.shape or
                    y_flat.shape == ()):
                # y_flat has to be broadcast over axes of x_flat[i]

                for i in take_idx.flatten():
                    if len(idx_[p:]) > 0:
                        x_flat_sub = x_flat[i].__getitem__(index)
                    else:
                        x_flat_sub = x_flat[i]
                    iadd(x_flat_sub, y_flat, broadcast=True)
            else:
                # y_flat's first axis corresponds to first exist of x_flat
                for j, i in enumerate(take_idx.flatten()):
                    if len(idx_[p:]) > 0:
                        x_flat_sub = x_flat[i].__getitem__(index)
                    else:
                        x_flat_sub = x_flat[i]
                    iadd(x_flat_sub, y_flat[j % y_flat.shape[0]], broadcast=True)
        x_ = x_flat.reshape(x_.shape).transpose(*rtransp)
        out[0] = x_


class GpuAdvancedIncSubtensor(HideC, BaseGpuAdvancedIncSubtensor, tensor.AdvancedIncSubtensor):
    """
    Implement AdvancedIncSubtensor on the gpu.

    """
    def make_node(self, x, y, *inputs):
        ctx_name = infer_context_name(x, y)
        rval = tensor.AdvancedIncSubtensor.make_node(self, x, y, *inputs)
        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
                             broadcastable=rval.outputs[0].type.broadcastable,
                             context_name=ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        return gof.Apply(self, [x, y] + rval.inputs[2:], [otype()])


class GpuAdvancedBooleanIncSubtensor(HideC, BaseGpuAdvancedIncSubtensor, tensor.AdvancedBooleanIncSubtensor):
    """
    Implement AdvancedBooleanIncSubtensor on the gpu.

    """
    def make_node(self, x, y, *inputs):
        ctx_name = infer_context_name(x, y)
        rval = tensor.AdvancedBooleanIncSubtensor.make_node(self, x, y, *inputs)
        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
                             broadcastable=rval.outputs[0].type.broadcastable,
                             context_name=ctx_name)
        x = as_gpuarray_variable(x, ctx_name)
        y = as_gpuarray_variable(y, ctx_name)
        return gof.Apply(self, [x, y] + rval.inputs[2:], [otype()])


class GpuAdvancedIncSubtensor1(Op):
    """
    Implement AdvancedIncSubtensor1 on the gpu.

    """
    _f16_ok = True
    __props__ = ('inplace', 'set_instead_of_inc')
    params_type = ParamsType(inplace=bool_t,
                             set_instead_of_inc=bool_t,
                             context=gpu_context_type,
                             # following params are used into c_init_code_struct(),
                             # as inputs are not available in that function.
                             ndim_input_0=size_t,
                             ndim_input_1=size_t,
                             typecode_input_0=int_t,
                             typecode_input_1=int_t)

    def __init__(self, inplace=False, set_instead_of_inc=False):
        self.inplace = inplace
        self.set_instead_of_inc = set_instead_of_inc
        if inplace:
            self.destroy_map = {0: [0]}

    def clone_inplace(self):
        return self.__class__(
            inplace=True,
            set_instead_of_inc=self.set_instead_of_inc)

    def make_node(self, x, y, ilist):
        ctx_name = infer_context_name(x, y)
        x_ = as_gpuarray_variable(x, ctx_name)
        y_ = as_gpuarray_variable(y, ctx_name)
        ilist_ = tensor.as_tensor_variable(ilist)

        assert x_.type.ndim >= y_.type.ndim

        if ilist_.type.dtype not in tensor.integer_dtypes:
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = 'set'
            else:
                opname = 'increment'
            raise TypeError(
                'cannot %s x subtensor with ndim=%s by y with ndim=%s ' % (
                    opname, x_.type.ndim, y_.type.ndim))

        return gof.Apply(self, [x_, y_, ilist_], [x_.type()])

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.outputs[0].type.context,
                                           # following params are used into c_init_code_struct().
                                           ndim_input_0=node.inputs[0].ndim,
                                           ndim_input_1=node.inputs[1].ndim,
                                           typecode_input_0=node.inputs[0].type.typecode,
                                           typecode_input_1=node.inputs[1].type.typecode)

    # We can't use the parent version that loops on each index
    # as we also need to loop when set_instead_of_inc is True and the
    # parent doesn't loop in that case.
    def perform(self, node, inp, out_, params=None):
        # TODO opt to make this inplace
        x, y, idx = inp
        out, = out_

        if not self.inplace:
            x = x.copy()

        out[0] = x

        if len(idx) == 0:
            return

        # Make sure idx is not a GpuArray otherwise we cannot use its
        # content to index x and y (This is because we serve as
        # fallback for _dev20).
        if isinstance(idx, gpuarray.GpuArray):
            idx = np.asarray(idx)

        # If `y` has as many dimensions as `x`, then we want to iterate
        # jointly on `x` and `y`. Otherwise, it means `y` should be
        # broadcasted to fill all relevant rows of `x`.
        if y.ndim == x.ndim and y.shape[0] != 1:
            assert len(y) == len(idx)
            if self.set_instead_of_inc:
                for (j, i) in enumerate(idx):
                    x[i] = y[j]
            else:
                k = get_iadd(node.inputs[0], node.inputs[1])
                for (j, i) in enumerate(idx):
                    k(x[i], y[j], broadcast=True)
        else:
            if y.ndim == x.ndim:
                # First dim is always 1 in this case.
                reshaped_y = y.reshape(y.shape[1:])
            else:
                nb_dims_to_add = (x.ndim - 1) - y.ndim
                reshaped_y = y.reshape((1,) * nb_dims_to_add + y.shape)

            if self.set_instead_of_inc:
                for i in idx:
                    x[i] = reshaped_y
            else:
                k = get_iadd(node.inputs[0], node.inputs[1])
                for i in idx:
                    k(x[i], reshaped_y, broadcast=True)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/error.h>', '<gpuarray/array.h>',
                '<gpuarray/elemwise.h>', 'gpuarray_helper.h']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_support_code_struct(self, node, nodename):
        return "\nGpuElemwise *iadd;\n"

    def c_init_code_struct(self, node, name, sub):
        return """
        gpuelemwise_arg args[2] = {{0}};
        args[0].name = "a";
        args[0].typecode = %(params)s->typecode_input_0;
        args[0].flags = GE_READ|GE_WRITE;
        args[1].name = "b";
        args[1].typecode = %(params)s->typecode_input_1;
        args[1].flags = GE_READ;
        iadd = GpuElemwise_new(%(params)s->context->ctx, "", "a += b",
                               2, args, %(params)s->ndim_input_1, GE_CONVERT_F16);
        if (iadd == NULL) {
          PyErr_SetString(PyExc_RuntimeError, "Could not intialize inplace add support");
          %(fail)s
        }
        """ % dict(params=sub['params'], fail=sub['fail'])

    def c_code(self, node, name, inputs, outputs, sub):
        if (node.inputs[0].ndim != node.inputs[1].ndim):
            raise NotImplementedError("This case does not have C code yet.")

        return """
        PyGpuArrayObject *row_x, *row_y;
        size_t nd = %(params)s->ndim_input_0;
        ssize_t *start = NULL, *step = NULL;
        size_t num_indices, j;
        int ret;
        int broadcast_y;

        start = (ssize_t*)malloc(nd * sizeof(ssize_t));
        step = (ssize_t*)malloc(nd * sizeof(ssize_t));
        if (start == NULL || step == NULL) {
            PyErr_NoMemory();
            %(fail)s
        }

        for (j = 0; j < nd; ++j) {
          start[j] = 0;
          step[j] = 1;
        }
        step[0] = 0;
        num_indices = PyArray_SIZE(%(ind)s);
        if (!%(params)s->inplace) {
          %(out)s = theano_try_copy(%(out)s, %(x)s);
          if (%(out)s == NULL) {
            // Exception already set
            %(fail)s
            }
        } else {
          Py_XDECREF(%(out)s);
          %(out)s = %(x)s;
          Py_INCREF(%(out)s);
        }
        if (num_indices != 0) {
          if ((num_indices - 1) > LONG_MAX) {
            PyErr_Format(PyExc_AssertionError,
                         "num_indices %%lld exceeds LONG_MAX + 1", (long long)num_indices);
            %(fail)s
          }
          broadcast_y = PyGpuArray_DIM(%(y)s, 0) == 1;
          for (j = 0; j < num_indices; j++) {
            start[0] = *(dtype_%(ind)s *)PyArray_GETPTR1(%(ind)s, j);
            if (start[0] < 0)
              start[0] += PyGpuArray_DIM(%(out)s, 0);
            if (start[0] < 0 || start[0] >= PyGpuArray_DIM(%(out)s, 0)) {
               PyErr_SetString(PyExc_IndexError, "index out of bounds");
               %(fail)s;
            }
            row_x = pygpu_index(%(out)s, start, (ssize_t *)PyGpuArray_DIMS(%(out)s), step);
            if (row_x == NULL)
              %(fail)s;

            if (broadcast_y)
              start[0] = 0;
            else
              start[0] = j;

            row_y = pygpu_index(%(y)s, start, (ssize_t *)PyGpuArray_DIMS(%(y)s), step);
            if (row_y == NULL) {
              Py_DECREF(row_x);
              %(fail)s;
            }

            if (%(params)s->set_instead_of_inc) {
              ret = GpuArray_setarray(&row_x->ga, &row_y->ga);
            } else {
              void *args[2];
              args[0] = (void *)&row_x->ga;
              args[1] = (void *)&row_y->ga;
              ret = GpuElemwise_call(iadd, args, GE_BROADCAST | GE_PADSHAPE);
            }
            Py_DECREF(row_x);
            Py_DECREF(row_y);
            if (ret != GA_NO_ERROR)
              PyErr_SetString(PyExc_RuntimeError, "Failed to set/inc elements");
          }
        }

        free(start);
        free(step);
        """ % dict(x=inputs[0], y=inputs[1], ind=inputs[2], out=outputs[0],
                   params=sub['params'],
                   fail="""
                   {
                        free(start);
                        free(step);
                        %(fail)s
                   }
                   """ % dict(fail=sub['fail']))

    def c_code_cache_version(self):
        return (5,)


class GpuAdvancedIncSubtensor1_dev20(GpuKernelBase, HideC,
                                     GpuAdvancedIncSubtensor1):
    """
    Implement AdvancedIncSubtensor1 on the gpu with atomics

    """
    _f16_ok = True
    params_type = GpuAdvancedIncSubtensor1.params_type
    get_params = GpuAdvancedIncSubtensor1.get_params

    def make_node(self, x, y, ilist):
        """
        It differs from GpuAdvancedIncSubtensor1 in that it makes sure
        the indexes are of type long.

        """
        ctx_name = infer_context_name(x, y, ilist)
        x_ = as_gpuarray_variable(x, ctx_name)
        y_ = as_gpuarray_variable(y.astype(x.dtype), ctx_name)
        ilist_ = as_gpuarray_variable(ilist, ctx_name)

        assert x_.type.ndim >= y_.type.ndim

        if ilist_.type.dtype not in tensor.integer_dtypes:
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = 'set'
            else:
                opname = 'increment'
            raise TypeError(
                'cannot %s x subtensor with ndim=%s by y with ndim=%s ' % (
                    opname, x_.type.ndim, y_.type.ndim))

        return gof.Apply(self, [x_, y_, ilist_], [x_.type()])

    def perform(self, node, inp, out, params):
        return super(GpuAdvancedIncSubtensor1_dev20, self).perform(node, inp, out)

    def c_code_cache_version(self):
        return (14,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray_helper.h>',
                '<gpuarray/types.h>']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def c_code(self, node, name, inputs, outputs, sub):
        if (node.inputs[0].ndim != node.inputs[1].ndim or
                node.inputs[0].ndim != 2):
            raise NotImplementedError("This case does not have C code yet.")

        return """
int err;
if (%(params)s->inplace) {
  Py_XDECREF(%(out)s);
  %(out)s = %(x)s;
  Py_INCREF(%(out)s);
} else {
  %(out)s = theano_try_copy(%(out)s, %(x)s);
}
if (!%(out)s) {
  // Exception already set
  %(fail)s
}
if (GpuArray_vector_add_fast(%(out)s, %(y)s, %(ind)s, %(params)s->set_instead_of_inc)) {
  %(fail)s
}
        """ % dict(x=inputs[0], y=inputs[1], ind=inputs[2], out=outputs[0], fail=sub['fail'], params=sub['params'])

    def gpu_kernels(self, node, nodename):
        # We can't rely on numpy for this, it changes with the OS
        CHARMAP = dict(int32='i', uint32='I',
                       int64='l', uint64='L',
                       float16='e', float32='f', float64='d')
        dtype_x = node.inputs[0].dtype
        dtype_y = node.inputs[1].dtype
        dtype_ind = node.inputs[2].dtype
        type_x = gpuarray.dtype_to_ctype(dtype_x)
        type_y = gpuarray.dtype_to_ctype(dtype_y)
        type_ind = gpuarray.dtype_to_ctype(dtype_ind)
        flags = Kernel.get_flags(dtype_x, dtype_y, dtype_ind)
        kname = "k_vector_add_fast"
        k_var = "k_vector_add_fast_" + nodename
        code = """#include "cluda.h"
        KERNEL void k_vector_add_fast(const ga_size numRowsX,
                                      const ga_size numColsX,
                                      const ga_ssize stridesX0,
                                      const ga_ssize stridesX1,
                                      GLOBAL_MEM %(type_x)s *X,
                                      const ga_size offset_X,
                                      const ga_size numRowsY,
                                      const ga_size numColsY,
                                      const ga_ssize stridesY0,
                                      const ga_ssize stridesY1,
                                      GLOBAL_MEM %(type_y)s *Y,
                                      const ga_size offset_Y,
                                      const ga_size numIndices,
                                      const ga_ssize stridesIndices,
                                      GLOBAL_MEM %(type_ind)s *indices_arr,
                                      const ga_size offset_indices_arr,
                                      const ga_int set_instead_of_inc,
                                      GLOBAL_MEM ga_int *err)
        {
             X = (GLOBAL_MEM %(type_x)s *)(((GLOBAL_MEM char *)X)+offset_X);
             Y = (GLOBAL_MEM %(type_y)s *)(((GLOBAL_MEM char *)Y)+offset_Y);
             indices_arr = (GLOBAL_MEM %(type_ind)s *)(((GLOBAL_MEM char *)indices_arr)+offset_indices_arr);

             for (ga_int i = GID_0; i < numIndices; i += GDIM_0)
             {
                  for (ga_int j = LID_0; j < numColsX; j += LDIM_0)
                  {
                      ga_ssize x_row = indices_arr[i * stridesIndices];
                      if (x_row < 0)
                          x_row += numRowsX;
                      ga_ssize y_row = i;
                      if (x_row < numRowsX && x_row >= 0) {
                        if (set_instead_of_inc) {
                          atom_xchg_%(tc)sg(&X[(x_row * stridesX0) + (j * stridesX1)],
                                   Y[(y_row * stridesY0) + (j * stridesY1)]);
                        } else {
                          atom_add_%(tc)sg(&X[(x_row * stridesX0) + (j * stridesX1)],
                                    Y[(y_row * stridesY0) + (j * stridesY1)]);
                        }
                      } else {
                        *err = 1;
                      }
                  }
             }
             return;
        }
        """ % dict(type_x=type_x, type_y=type_y, type_ind=type_ind,
                   tc=CHARMAP[dtype_x])
        from pygpu.gpuarray import SIZE, SSIZE
        params = [
            SIZE, SIZE, SSIZE, SSIZE, gpuarray.GpuArray, SIZE,
            SIZE, SIZE, SSIZE, SSIZE, gpuarray.GpuArray, SIZE,
            SIZE, SSIZE, gpuarray.GpuArray, SIZE, 'int32',
            gpuarray.GpuArray]
        return [Kernel(code=code, name=kname, params=params,
                       flags=flags, objvar=k_var)]

    def c_support_code_struct(self, node, nodename):
        return super(GpuAdvancedIncSubtensor1_dev20, self).c_support_code_struct(node, nodename) + """
        int GpuArray_vector_add_fast(PyGpuArrayObject* py_self,
                                     PyGpuArrayObject* py_other,
                                     PyGpuArrayObject* indices_arr,
                                     const int set_instead_of_inc)
        {
            size_t threads_per_block = std::min(PyGpuArray_DIMS(py_self)[1], (size_t)256);
            size_t n_blocks = std::min(PyGpuArray_SIZE(indices_arr), (size_t)4096);
            gpudata *errbuf;
            int err, kerr = 0;
            size_t itemsize_x = GpuArray_ITEMSIZE(&py_self->ga);
            size_t itemsize_y = GpuArray_ITEMSIZE(&py_other->ga);
            size_t itemsize_ind = GpuArray_ITEMSIZE(&indices_arr->ga);

            if (threads_per_block > 0 && n_blocks > 0) {
              err = gpudata_property(py_self->ga.data,
                                     GA_CTX_PROP_ERRBUF, &errbuf);
              if (err != GA_NO_ERROR) {
                PyErr_SetString(PyExc_RuntimeError, "Can't fetch error buffer");
                return 1;
              }

              err = k_vector_add_fast_call(
        1, &n_blocks, &threads_per_block, 0,
        PyGpuArray_DIMS(py_self)[0],
        PyGpuArray_DIMS(py_self)[1],
        PyGpuArray_STRIDES(py_self)[0] / itemsize_x,
        PyGpuArray_STRIDES(py_self)[1] / itemsize_x,
        py_self->ga.data,
        py_self->ga.offset,
        PyGpuArray_DIMS(py_other)[0],
        PyGpuArray_DIMS(py_other)[1],
        PyGpuArray_DIMS(py_other)[0] == 1 ? 0 : PyGpuArray_STRIDES(py_other)[0] / itemsize_y,
        PyGpuArray_DIMS(py_other)[1] == 1 ? 0 : PyGpuArray_STRIDES(py_other)[1] / itemsize_y,
        py_other->ga.data,
        py_other->ga.offset,
        PyGpuArray_DIMS(indices_arr)[0],
        PyGpuArray_STRIDES(indices_arr)[0] / itemsize_ind,
        indices_arr->ga.data,
        indices_arr->ga.offset,
        set_instead_of_inc,
        errbuf);

              if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                return 1;
              }
              err = gpudata_read(&kerr, errbuf, 0, sizeof(int));
              if (err != GA_NO_ERROR) {
                PyErr_SetString(PyExc_RuntimeError, "Can't read error buffer");
                return 1;
              }
              if (kerr != 0) {
                PyErr_SetString(PyExc_IndexError, "Index out of bounds");
                kerr = 0;
                gpudata_write(errbuf, 0, &kerr, sizeof(int));
                return 1;
              }
            }
          return 0;
        }
        """ % dict(k_var="k_vector_add_fast_" + nodename)


class GpuExtractDiag(Op):
    __props__ = ("offset", "axis1", "axis2", "view")
    _f16_ok = True

    def __init__(self, offset=0, axis1=0, axis2=1, view=False):
        self.view = view
        if self.view:
            self.view_map = {0: [0]}
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def make_node(self, _x):
        ctx_name = infer_context_name(_x)
        x = as_gpuarray_variable(_x, ctx_name)

        if x.ndim < 2:
            raise ValueError('Diagonal needs an input with 2 or more '
                             'dimensions', x)
        axis_small, axis_large = sorted((self.axis1, self.axis2))
        broadcastable = x.broadcastable[:axis_small] + \
            x.broadcastable[axis_small + 1:axis_large] + \
            x.broadcastable[axis_large + 1:] + (False,)
        return gof.Apply(self, [x], [x.type.clone(broadcastable=broadcastable)()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        # zero-dimensional matrices ...
        if x.size == 0:
            out_shape = [d for i, d in enumerate(x.shape)
                         if i not in (self.axis1, self.axis2)]
            diag_size = np.min((x.shape[self.axis1], x.shape[self.axis2]))
            out_shape.append(diag_size)
            z[0] = node.outputs[0].type.value_zeros(tuple(out_shape))
            return

        # step 1) slicing on axis1 and axis2.
        if self.offset >= 0:
            stride_axis, slice_axis = self.axis1, self.axis2
        else:
            slice_axis, stride_axis = self.axis1, self.axis2

        small_axis, large_axis = sorted((x.shape[self.axis1],
                                         x.shape[self.axis2]))

        if x.shape[stride_axis] < x.shape[slice_axis]:
            # in the bigger triangle
            numstride = small_axis - np.max((
                0, small_axis + np.abs(self.offset) - large_axis))
        else:
            # in the smaller triangle
            numstride = small_axis - np.abs(self.offset)

        slicer = [np.s_[:], ] * x.ndim
        slicer[stride_axis] = np.s_[:numstride]
        slicer[slice_axis] = np.abs(self.offset)
        slicer = tuple(slicer)

        # step 2) Swap stride_axis to the last dim because we want the dim on
        # which the diags extracted be listed as the last dim of the tensor.
        # This is also in consistence with the interface of numpy.diagonal.
        if slice_axis < stride_axis:
            stride_axis -= 1
        new_dim_order = list(range(x[slicer].ndim))
        new_dim_order = tuple(new_dim_order[:stride_axis] +
                              new_dim_order[stride_axis + 1:] +
                              [stride_axis, ])
        rval = x[slicer].transpose(new_dim_order)

        # step 3) modify the strides in the last axis, such that rval becomes
        # a view on the diagonal.
        other_strides = tuple([d for i, d in enumerate(x.strides)
                               if i not in (self.axis1, self.axis2)])
        rval.strides = other_strides + \
            (x.strides[self.axis1] + x.strides[self.axis2], )

        if self.view:
            z[0] = rval
        else:
            z[0] = rval.copy()

    def grad(self, inputs, gout):
        (input_x,) = inputs
        return [grad_not_implemented(self, 0, input_x)]

    def infer_shape(self, node, shapes):
        in_shape, = shapes
        dim1 = in_shape[self.axis1]
        dim2 = in_shape[self.axis2]
        out_shape = [d for i, d in enumerate(in_shape)
                     if i not in (self.axis1, self.axis2)]
        # The following logic is inspired by C code of PyArray_Diagonal().
        offset = self.offset
        if offset > 0:
            diag_size = T.clip(dim2 - offset, 0, dim1)
        elif offset < 0:
            diag_size = T.clip(dim1 + offset, 0, dim2)
        else:
            diag_size = T.minimum(dim1, dim2)
        out_shape.append(diag_size)
        return [tuple(out_shape)]


class GpuAllocDiag(AllocDiag):
    __props__ = ("offset", "axis1", "axis2")

    def make_node(self, diag):
        ctx_name = infer_context_name(diag)
        diag = as_gpuarray_variable(diag, ctx_name)
        if diag.type.ndim < 1:
            raise ValueError('AllocDiag needs an input with 1 or more '
                             'dimensions', diag.type)
        return gof.Apply(
            self, [diag],
            [diag.type.__class__(
                dtype=diag.dtype,
                broadcastable=[False] * (diag.ndim + 1))()]
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        axis1 = np.minimum(self.axis1, self.axis2)
        axis2 = np.maximum(self.axis1, self.axis2)
        offset = self.offset

        # Initialise a buffer the same size as the output
        result_shape = x.shape[:-1] + (x.shape[-1] + abs(offset),) * 2
        result_buffer_shape = ((np.prod(x.shape[:-1]).astype(np.int64),) +
                               (x.shape[-1] + abs(offset),) * 2)
        result_buffer = gpuarray.zeros(result_buffer_shape,
                                       dtype=x.dtype,
                                       context=x.context)

        # Slice out a view of the diagonals
        if offset < 0:  # diag in the lower triangle
            diag_view = result_buffer[:, abs(offset):, 0]
        else:  # diag in the upper triangle
            diag_view = result_buffer[:, :x.shape[-1], abs(offset)]
        diag_view.strides = (diag_view.strides[0],
                             diag_view.strides[1] + x.dtype.itemsize)

        # Fill view with flattened array of diagonals
        diag_view[:] = x.reshape(diag_view.shape)[:]

        # Unflatten buffer into output size
        result = result_buffer.reshape(result_shape)

        if len(x.shape) > 1:
            # Re-order axes so they correspond to diagonals at axis1, axis2
            axes = list(range(len(x.shape[:-1])))
            last_idx = axes[-1]
            axes = axes[:axis1] + [last_idx + 1] + axes[axis1:]
            axes = axes[:axis2] + [last_idx + 2] + axes[axis2:]
            result = result.transpose(axes)

        z[0] = result

    def grad(self, inputs, gout):
        (gz,) = gout
        return [GpuExtractDiag(offset=self.offset, axis1=self.axis1, axis2=self.axis2)(gz)]
