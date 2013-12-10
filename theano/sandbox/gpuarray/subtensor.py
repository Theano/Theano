import StringIO

import numpy

import theano
from theano import tensor, gof
from theano.tensor.subtensor import IncSubtensor, Subtensor, get_idx_list

from theano.gof.python25 import all, any

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from theano.sandbox.gpuarray.type import GpuArrayType
from theano.sandbox.gpuarray.basic_ops import as_gpuarray_variable, HideC


class GpuSubtensor(HideC, Subtensor):
    def make_node(self, x, *inputs):
        rval = tensor.Subtensor.make_node(self, x, *inputs)
        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
                             broadcastable=rval.outputs[0].type.broadcastable)
        x = as_gpuarray_variable(x)
        return gof.Apply(self, [x] + rval.inputs[1:], [otype()])

    def perform(self, node, inputs, out_):
        out, = out_
        x = inputs[0]
        if self.perform_cache_cdata is not None:
            out[0] = x.__getitem__(self.perform_cache_cdata)
            return

        cdata = get_idx_list(inputs, self.idx_list)
        if len(cdata) == 1:
            cdata = cdata[0]
        if len(inputs) == 1:
            self.perform_cache_cdata = cdata

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
                if (*start >= len) *start = (*step < 0) ? len-1 : len;
            }

            if (stop_n) *stop = (*step < 0) ? -1 : len;
            else {
                if (*stop < 0) *stop += len;
                if (*stop < 0) *stop = (*step < 0) ? -1 : 0;
                if (*stop >= len) *stop = (*step < 0) ? len-1 : len;
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
        if (!%(out)s) { %(fail)s }
""" % dict(out=outputs[0], inp=inp, fail=sub['fail'])

        sio = StringIO.StringIO()
        print >> sio, """
        ssize_t starts[%(sz)s];
        ssize_t stops[%(sz)s];
        ssize_t steps[%(sz)s];
        ssize_t cur;
        int err;

        if (%(inp)s->ga.nd != %(sz)s) {
            PyErr_SetString(PyExc_IndexError, "invalid index");
            %(fail)s
        }
        """ % dict(sz=len(idx_list), inp=inp, fail=sub['fail'])

        def fix_idx(idx):
            if idx is None:
                return "0", 1
            elif isinstance(idx, (numpy.integer, int)):
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
                print >>sio, """
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
                           fail=sub['fail'], inp=inp)
            else:
                if isinstance(idx, gof.Type):
                    start = indices.pop(0)
                elif isinstance(idx, (numpy.integer, int)):
                    start = idx
                else:
                    assert 0, idx
                print >>sio, """
                cur = %(start)s;
                if (cur < 0)
                    cur += %(inp)s->ga.dimensions[%(i)s];
                starts[%(i)s] = cur;
                steps[%(i)s] = 0;
                """ % dict(i=i, start=start, fail=sub['fail'], inp=inp)

        print >>sio, """
        Py_XDECREF(%(out)s);
        %(out)s = pygpu_index(%(inp)s, starts, stops, steps);
        if (!%(out)s) { %(fail)s }
""" % dict(name=name, fail=sub['fail'], inp=inp, out=outputs[0])

        return sio.getvalue()

    def c_code_cache_version(self):
        return (5,)


class GpuIncSubtensor(HideC, IncSubtensor):
    """
    Implement IncSubtensor on the gpu.

    Note: The optimization to make this inplace is in tensor/opt.
          The same optimization handles IncSubtensor and GpuIncSubtensor.
          This Op has c_code too; it inherits tensor.IncSubtensor's c_code.
          The helper methods like do_type_checking, copy_of_x, etc. specialize
          the c_code for this Op.
    """

    def make_node(self, x, y, *inputs):
        x = as_gpuarray_variable(x)
        y = as_gpuarray_variable(y)
        rval = tensor.IncSubtensor.make_node(self, x, y, *inputs)
        return gof.Apply(self, [x, y] + rval.inputs[2:], [x.type()])

    def perform(self, node, inputs, out_):
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
                #sub_x += y
                pygpu.elemwise.ielemwise2(sub_x, '+', y,  broadcast=False)
            else:
                #sub_x += -sub_x + y
                x.__setitem__(cdata, y)
        else:
            # scalar case
            if not self.set_instead_of_inc:
                #x.__setitem__(cdata, sub_x + y)
                tmp = pygpu.elemwise.elemwise2(sub_x, '+', y,  sub_x, broadcast=False)
                x.__setitem__(cdata, tmp)
            else:
                x.__setitem__(cdata, y)
        out[0] = x

    def do_type_checking(self, node):
        """ Should raise NotImplementedError if c_code does not support
        the types involved in this node.
        """

        if not isinstance(node.inputs[0].type, GpuArrayType):
            raise NotImplementedError()

    def copy_of_x(self, x):
        """
            :param x: a string giving the name of a C variable
                pointing to an array

            :return: C code expression to make a copy of x

            Base class uses `PyArrayObject *`, subclasses may override for
            different types of arrays.
        """
        return """pygpu_copy(%(x)s, GA_ANY_ORDER)""" % locals()

    def decl_view(self):
        return "PyGpuArray* zview = NULL;"

    def make_view_array(self, x, view_ndim):
        """//TODO
            :param x: a string identifying an array to be viewed
            :param view_ndim: a string specifying the number of dimensions
                to have in the view

            This doesn't need to actually set up the view with the
            right indexing; we'll do that manually later.
        """
        ret = """zview = (CudaNdarray*) CudaNdarray_New(%(view_ndim)s);
        if (CudaNdarray_set_device_data(
                zview,
                CudaNdarray_DEV_DATA(%(x)s) + xview_offset/4,
                (PyObject*) %(x)s))
        {
            zview = NULL;
            PyErr_Format(PyExc_RuntimeError,
                         "GpuSubtensor is not able to set the"
                         " devdata field of the view");
        }else{
            cnda_mark_dev_structure_dirty(zview);
            for(int idx=0;idx <%(view_ndim)s; idx++){
                if(xview_dims[idx]==1)
                    CudaNdarray_set_stride(zview, idx, 0);
                else
                    CudaNdarray_set_stride(zview, idx, xview_strides[idx]);
                CudaNdarray_set_dim(zview, idx, xview_dims[idx]);
            }
        }
        """ % locals()
        return ret

    def get_helper_c_code_args(self):
        """ Return a dictionary of arguments to use with helper_c_code"""
        return {'c_prefix': 'PyGpuArray',
                'strides_mul': 1
                }

    def copy_into(self, view, source):
        """
            view: string, C code expression for an array
            source: string, C code expression for an array

            returns a C code expression to copy source into view, and
            return 0 on success
        """
        return """GpuArray_move(%(view)s, %(source)s)""" % locals()

    def set_view_base(self, x, fail):
        return """//TODO
        //Set the base only now

        if(CudaNdarray_set_device_data(zview, CudaNdarray_DEV_DATA(zview),
                                    %(x)s)){
            PyErr_Format(PyExc_RuntimeError,
                         "GpuSubtensor is not able to set"
                         " the base of the view array");
            Py_XDECREF(zview);
            %(fail)s;
        }""" % locals()

    def add_to_zview(self, x, fail):
        #TODO
        return """
        PyObject * add_result = CudaNdarray_inplace_add((PyObject *) zview,
                                                        (PyObject *) py_%(x)s);

        if (! add_result )
        {
            Py_DECREF(zview);
            %(fail)s;
        }
        else
        {
            Py_DECREF(add_result);
        }
        """ % locals()

    def c_code_cache_version(self):
        return ()
        parent_version = super(GpuIncSubtensor, self).c_code_cache_version()
        if parent_version:
            return parent_version + (0,)
        return ()
