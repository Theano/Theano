#include <Python.h>
#include "theano_mod_helper.h"
#include "structmember.h"
#include <sys/time.h>

#if PY_VERSION_HEX >= 0x03000000
#include "numpy/npy_3kcompat.h"
#define PyCObject_AsVoidPtr  NpyCapsule_AsVoidPtr
#define PyCObject_GetDesc  NpyCapsule_GetDesc
#define PyCObject_Check NpyCapsule_Check
#endif

#ifndef Py_TYPE
#define Py_TYPE(obj) obj->ob_type
#endif

/**

TODO: 
- Check max supported depth of recursion
- CLazyLinker should add context information to errors caught during evaluation. Say what node we were on, add the traceback attached to the node.
- Clear containers of fully-useed intermediate results if allow_gc is 1
- Add timers for profiling
- Add support for profiling space used.


  */
static double pytime(const struct timeval * tv)
{
  struct timeval t;
  if (!tv)
    {
      tv = &t;
      gettimeofday(&t, NULL);
    }
  return (double) tv->tv_sec + (double) tv->tv_usec / 1000000.0;
}

/**
  Helper routine to convert a PyList of integers to a c array of integers.
  */
static int unpack_list_of_ssize_t(PyObject * pylist, Py_ssize_t **dst, Py_ssize_t *len,
                                  const char* kwname)
{
  Py_ssize_t buflen, *buf;
  if (!PyList_Check(pylist))
    {
      PyErr_Format(PyExc_TypeError, "%s must be list", kwname);
      return -1;
    }
  assert (NULL == *dst);
  *len = buflen = PyList_Size(pylist);
  *dst = buf = (Py_ssize_t*)calloc(buflen, sizeof(Py_ssize_t));
  assert(buf);
  for (int ii = 0; ii < buflen; ++ii)
    {
      PyObject * el_i = PyList_GetItem(pylist, ii);
      Py_ssize_t n_i = PyNumber_AsSsize_t(el_i, PyExc_IndexError);
      if (PyErr_Occurred())
        {
          free(buf);
          *dst = NULL;
          return -1;
        }
      buf[ii] = n_i;
    }
  return 0;
}

/**

  CLazyLinker


  */
typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    PyObject * nodes; // the python list of nodes
    PyObject * thunks; // python list of thunks
    PyObject * pre_call_clear; //list of cells to clear on call.
    int allow_gc;
    Py_ssize_t n_applies;
    int n_vars;    // number of variables in the graph
    int * var_computed; // 1 or 0 for every variable
    PyObject ** var_computed_cells;
    PyObject ** var_value_cells;
    Py_ssize_t **dependencies; // list of vars dependencies for GC
    Py_ssize_t *n_dependencies;

    Py_ssize_t n_output_vars;
    Py_ssize_t * output_vars; // variables that *must* be evaluated by call

    int * is_lazy; // 1 or 0 for every thunk

    Py_ssize_t * var_owner; // nodes[[var_owner[var_idx]]] is var[var_idx]->owner
    int * var_has_owner; //  1 or 0

    Py_ssize_t * node_n_inputs;
    Py_ssize_t * node_n_outputs;
    Py_ssize_t ** node_inputs;
    Py_ssize_t ** node_outputs;
    Py_ssize_t * node_inputs_outputs_base; // node_inputs and node_outputs point into this
    Py_ssize_t * node_n_prereqs;
    Py_ssize_t ** node_prereqs;

    Py_ssize_t * update_storage; // input cells to update with the last outputs in output_vars
    Py_ssize_t n_updates;

    void ** thunk_cptr_fn;
    void ** thunk_cptr_data;
    PyObject * call_times;
    PyObject * call_counts;
    int do_timing;
    int need_update_inputs;
    int position_of_error; // -1 for no error, otw the index into `thunks` that failed.
} CLazyLinker;


static void
CLazyLinker_dealloc(PyObject* _self)
{
  CLazyLinker* self = (CLazyLinker *) _self;
  free(self->thunk_cptr_fn);
  free(self->thunk_cptr_data);

  free(self->is_lazy);

  free(self->update_storage);

  if (self->node_n_prereqs)
    {
      for (int i = 0; i < self->n_applies; ++i)
        {
          free(self->node_prereqs[i]);
        }
    }
  free(self->node_n_prereqs);
  free(self->node_prereqs);
  free(self->node_inputs_outputs_base);
  free(self->node_n_inputs);
  free(self->node_n_outputs);
  free(self->node_inputs);
  free(self->node_outputs);

  if (self->dependencies)
    {
      for (int i = 0; i < self->n_vars; ++i)
        {
          free(self->dependencies[i]);
        }
      free(self->dependencies);
      free(self->n_dependencies);
    }

  free(self->var_owner);
  free(self->var_has_owner);
  free(self->var_computed);
  if (self->var_computed_cells)
    {
      for (int i = 0; i < self->n_vars; ++i)
        {
          Py_DECREF(self->var_computed_cells[i]);
          Py_DECREF(self->var_value_cells[i]);
        }
    }
  free(self->var_computed_cells);
  free(self->var_value_cells);
  free(self->output_vars);

  Py_XDECREF(self->nodes);
  Py_XDECREF(self->thunks);
  Py_XDECREF(self->call_times);
  Py_XDECREF(self->call_counts);
  Py_XDECREF(self->pre_call_clear);
  Py_TYPE(self)->tp_free((PyObject*)self);
}
static PyObject *
CLazyLinker_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    CLazyLinker *self;

    self = (CLazyLinker *)type->tp_alloc(type, 0);
    if (self != NULL) {
      self->nodes = NULL;
      self->thunks = NULL;
      self->pre_call_clear = NULL;

      self->allow_gc = 1;
      self->n_applies = 0;
      self->n_vars = 0;
      self->var_computed = NULL;
      self->var_computed_cells = NULL;
      self->var_value_cells = NULL;
      self->dependencies = NULL;
      self->n_dependencies = NULL;

      self->n_output_vars = 0;
      self->output_vars = NULL;

      self->is_lazy = NULL;

      self->var_owner = NULL;
      self->var_has_owner = NULL;

      self->node_n_inputs = NULL;
      self->node_n_outputs = NULL;
      self->node_inputs = NULL;
      self->node_outputs = NULL;
      self->node_inputs_outputs_base = NULL;
      self->node_prereqs = NULL;
      self->node_n_prereqs = NULL;

      self->update_storage = NULL;
      self->n_updates = 0;

      self->thunk_cptr_data = NULL;
      self->thunk_cptr_fn = NULL;
      self->call_times = NULL;
      self->call_counts = NULL;
      self->do_timing = 0;

      self->need_update_inputs = 0;
      self->position_of_error = -1;
    }
    return (PyObject *)self;
}

static int
CLazyLinker_init(CLazyLinker *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {
      (char*)"nodes",
      (char*)"thunks",
      (char*)"pre_call_clear",
      (char*)"allow_gc",
      (char*)"call_counts",
      (char*)"call_times",
      (char*)"compute_map_list",
      (char*)"storage_map_list",
      (char*)"base_input_output_list",
      (char*)"node_n_inputs",
      (char*)"node_n_outputs",
      (char*)"node_input_offset",
      (char*)"node_output_offset",
      (char*)"var_owner",
      (char*)"is_lazy_list",
      (char*)"output_vars",
      (char*)"node_prereqs",
      (char*)"node_output_size",
      (char*)"update_storage",
      (char*)"dependencies",
      NULL};

    PyObject *compute_map_list=NULL,
             *storage_map_list=NULL,
             *base_input_output_list=NULL,
             *node_n_inputs=NULL,
             *node_n_outputs=NULL,
             *node_input_offset=NULL,
             *node_output_offset=NULL,
             *var_owner=NULL,
             *is_lazy=NULL,
             *output_vars=NULL,
             *node_prereqs=NULL,
             *node_output_size=NULL,
             *update_storage=NULL,
             *dependencies=NULL;

    assert(!self->nodes);
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "OOOiOOOOOOOOOOOOOOOO", kwlist,
                                      &self->nodes,
                                      &self->thunks,
                                      &self->pre_call_clear,
                                      &self->allow_gc,
                                      &self->call_counts,
                                      &self->call_times,
                                      &compute_map_list,
                                      &storage_map_list,
                                      &base_input_output_list,
                                      &node_n_inputs,
                                      &node_n_outputs,
                                      &node_input_offset,
                                      &node_output_offset,
                                      &var_owner,
                                      &is_lazy,
                                      &output_vars,
                                      &node_prereqs,
                                      &node_output_size,
                                      &update_storage,
                                      &dependencies
                                      ))
        return -1;
    Py_INCREF(self->nodes);
    Py_INCREF(self->thunks);
    Py_INCREF(self->pre_call_clear);
    Py_INCREF(self->call_counts);
    Py_INCREF(self->call_times);

    Py_ssize_t n_applies = PyList_Size(self->nodes);

    self->n_applies = n_applies;
    self->n_vars = PyList_Size(var_owner);

    if (PyList_Size(self->thunks) != n_applies) return -1;
    if (PyList_Size(self->call_counts) != n_applies) return -1;
    if (PyList_Size(self->call_times) != n_applies) return -1;

    // allocated and initialize thunk_cptr_data and thunk_cptr_fn
    if (n_applies)
      {
        self->thunk_cptr_data = (void**)calloc(n_applies, sizeof(void*));
        self->thunk_cptr_fn = (void**)calloc(n_applies, sizeof(void*));
        self->is_lazy = (int*)calloc(n_applies, sizeof(int));
        self->node_prereqs = (Py_ssize_t**)calloc(n_applies, sizeof(Py_ssize_t*));
        self->node_n_prereqs = (Py_ssize_t*)calloc(n_applies, sizeof(Py_ssize_t));
        assert(self->node_prereqs);
        assert(self->node_n_prereqs);
        assert(self->is_lazy);
        assert(self->thunk_cptr_fn);
        assert(self->thunk_cptr_data);

        for (int i = 0; i < n_applies; ++i)
          {
            PyObject * thunk = PyList_GetItem(self->thunks, i);
            //thunk is borrowed
            if (PyObject_HasAttrString(thunk, "cthunk"))
              {
                PyObject * cthunk = PyObject_GetAttrString(thunk, "cthunk");
                //new reference
                assert (cthunk && PyCObject_Check(cthunk));
                self->thunk_cptr_fn[i] = PyCObject_AsVoidPtr(cthunk);
                self->thunk_cptr_data[i] = PyCObject_GetDesc(cthunk);
                Py_DECREF(cthunk);
                // cthunk is kept alive by membership in self->thunks
              }

            PyObject * el_i = PyList_GetItem(is_lazy, i);
            self->is_lazy[i] = PyNumber_AsSsize_t(el_i, NULL);

            /* now get the prereqs */
            el_i = PyList_GetItem(node_prereqs, i);
            assert (PyList_Check(el_i));
            self->node_n_prereqs[i] = PyList_Size(el_i);
            if (self->node_n_prereqs[i])
              {
                self->node_prereqs[i] = (Py_ssize_t*)malloc(
                              PyList_Size(el_i)*sizeof(Py_ssize_t));
                for (int j = 0; j < PyList_Size(el_i); ++j)
                  {
                    PyObject * el_ij = PyList_GetItem(el_i, j);
                    Py_ssize_t N = PyNumber_AsSsize_t(el_ij, PyExc_IndexError);
                    if (PyErr_Occurred())
                      return -1;
                    // N < n. variables
                    assert(N < PyList_Size(var_owner));
                    self->node_prereqs[i][j] = N;
                  }
              }
          }
      }
    if (PyList_Check(base_input_output_list))
      {
        Py_ssize_t n_inputs_outputs_base = PyList_Size(base_input_output_list);
        self->node_inputs_outputs_base = (Py_ssize_t*)calloc(n_inputs_outputs_base,sizeof(Py_ssize_t));
        assert(self->node_inputs_outputs_base);
        for (int i = 0; i < n_inputs_outputs_base; ++i)
          {
            PyObject *el_i = PyList_GetItem(base_input_output_list, i);
            Py_ssize_t idx = PyNumber_AsSsize_t(el_i, PyExc_IndexError);
            if (PyErr_Occurred()) return -1;
            self->node_inputs_outputs_base[i] = idx;
          }
        self->node_n_inputs = (Py_ssize_t*)calloc(n_applies,sizeof(Py_ssize_t));
        assert(self->node_n_inputs);
        self->node_n_outputs = (Py_ssize_t*)calloc(n_applies,sizeof(Py_ssize_t));
        assert(self->node_n_outputs);
        self->node_inputs = (Py_ssize_t**)calloc(n_applies,sizeof(Py_ssize_t*));
        assert(self->node_inputs);
        self->node_outputs = (Py_ssize_t**)calloc(n_applies,sizeof(Py_ssize_t*));
        assert(self->node_outputs);
        for (int i = 0; i < n_applies; ++i)
          {
            Py_ssize_t N;
            N = PyNumber_AsSsize_t(PyList_GetItem(node_n_inputs, i),PyExc_IndexError);
            if (PyErr_Occurred()) return -1;
            assert (N <= n_inputs_outputs_base);
            self->node_n_inputs[i] = N;
            N = PyNumber_AsSsize_t(PyList_GetItem(node_n_outputs, i),PyExc_IndexError);
            if (PyErr_Occurred()) return -1;
            assert (N <= n_inputs_outputs_base);
            self->node_n_outputs[i] = N;
            N = PyNumber_AsSsize_t(PyList_GetItem(node_input_offset, i),PyExc_IndexError);
            if (PyErr_Occurred()) return -1;
            assert (N <= n_inputs_outputs_base);
            self->node_inputs[i] = &self->node_inputs_outputs_base[N];
            N = PyNumber_AsSsize_t(PyList_GetItem(node_output_offset, i),PyExc_IndexError);
            if (PyErr_Occurred()) return -1;
            assert (N <= n_inputs_outputs_base);
            self->node_outputs[i] = &self->node_inputs_outputs_base[N];
          }
      }
    else
      {
        PyErr_SetString(PyExc_TypeError, "base_input_output_list must be list");
        return -1;
      }

    // allocation for var_owner
    if (PyList_Check(var_owner))
      {
        self->var_owner = (Py_ssize_t*)calloc(self->n_vars,sizeof(Py_ssize_t));
        self->var_has_owner = (int*)calloc(self->n_vars,sizeof(int));
        self->var_computed = (int*)calloc(self->n_vars,sizeof(int));
        self->var_computed_cells = (PyObject**)calloc(self->n_vars,sizeof(PyObject*));
        self->var_value_cells = (PyObject**)calloc(self->n_vars,sizeof(PyObject*));
        for (int i = 0; i < self->n_vars; ++i)
          {
            PyObject * el_i = PyList_GetItem(var_owner, i);
            if (el_i == Py_None)
              {
                self->var_has_owner[i] = 0;
              }
            else
              {
                Py_ssize_t N = PyNumber_AsSsize_t(el_i, PyExc_IndexError);
                if (PyErr_Occurred()) return -1;
                assert (N <= n_applies);
                self->var_owner[i] = N;
                self->var_has_owner[i] = 1;
              }
            self->var_computed_cells[i] = PyList_GetItem(compute_map_list, i);
            Py_INCREF(self->var_computed_cells[i]);
            self->var_value_cells[i] = PyList_GetItem(storage_map_list, i);
            Py_INCREF(self->var_value_cells[i]);
          }
      }
    else
      {
        PyErr_SetString(PyExc_TypeError, "var_owner must be list");
        return -1;
      }

    if (dependencies != Py_None)
      {
        self->dependencies = (Py_ssize_t**)calloc(self->n_vars, sizeof(Py_ssize_t *));
        self->n_dependencies = (Py_ssize_t*)calloc(self->n_vars, sizeof(Py_ssize_t));
        assert(self->dependencies);
        assert(self->n_dependencies);

        for (int i = 0; i < self->n_vars; ++i)
          {
            PyObject *tmp = PyList_GetItem(dependencies, i);
            // refcounting - tmp is borrowed
            if (unpack_list_of_ssize_t(tmp, &self->dependencies[i], &self->n_dependencies[i],
                                       "dependencies"))
              return -1;
          }
      }

    if (unpack_list_of_ssize_t(output_vars, &self->output_vars, &self->n_output_vars,
                               "output_vars"))
      return -1;
    for (int i = 0; i < self->n_output_vars; ++i)
      {
        assert(self->output_vars[i] < self->n_vars);
      }
    if (unpack_list_of_ssize_t(update_storage, &self->update_storage, &self->n_updates,
                               "updates_storage"))
      return -1;
    return 0;
}
static void set_position_of_error(CLazyLinker * self, int owner_idx)
{
  if (self->position_of_error == -1)
    {
      self->position_of_error = owner_idx;
    }
}
static PyObject * pycall(CLazyLinker * self, Py_ssize_t node_idx, int verbose)
{
  // call thunk to see which inputs it wants
  PyObject * thunk = PyList_GetItem(self->thunks, node_idx);
  // refcounting - thunk is borrowed
  PyObject * rval = NULL;
  if (self->do_timing)
    {
      double t0 = pytime(NULL);
      if (verbose) fprintf(stderr, "calling via Python (node %i)\n", (int)node_idx);
      rval = PyObject_CallObject(thunk, NULL);
      if (rval)
        {
          double t1 = pytime(NULL);
          double ti = PyFloat_AsDouble(
                         PyList_GetItem(self->call_times, node_idx));
          PyList_SetItem(self->call_times, node_idx,
                         PyFloat_FromDouble(t1 - t0 + ti));
          PyObject * count = PyList_GetItem(self->call_counts, node_idx);
          long icount = PyInt_AsLong(count);
          PyList_SetItem(self->call_counts, node_idx,
                         PyInt_FromLong(icount + 1));
      }
    }
  else
    {
      if (verbose)
        {
          fprintf(stderr, "calling via Python (node %i)\n", (int)node_idx);
        }
      rval = PyObject_CallObject(thunk, NULL);
    }
  return rval;
}
static int c_call(CLazyLinker * self, Py_ssize_t node_idx, int verbose)
{
  void * ptr_addr = self->thunk_cptr_fn[node_idx];
  int (*fn)(void*) = (int (*)(void*))(ptr_addr);
  if (verbose) fprintf(stderr, "calling non-lazy shortcut (node %i)\n", (int)node_idx);
  int err = 0;
  if (self->do_timing)
    {
      double t0 = pytime(NULL);
      err = fn(self->thunk_cptr_data[node_idx]);
      double t1 = pytime(NULL);
      double ti = PyFloat_AsDouble(PyList_GetItem(self->call_times, node_idx));
      PyList_SetItem(self->call_times, node_idx, PyFloat_FromDouble(t1 - t0 + ti));
      PyObject * count = PyList_GetItem(self->call_counts, node_idx);
      long icount = PyInt_AsLong(count);
      PyList_SetItem(self->call_counts, node_idx, PyInt_FromLong(icount+1));
    }
  else
    {
      err = fn(self->thunk_cptr_data[node_idx]);
    }

  if (err)
    {
      // cast the argument to a PyList (as described near line 226 of cc.py)
      PyObject * __ERROR = ((PyObject**)self->thunk_cptr_data[node_idx])[0];
      assert (PyList_Check(__ERROR));
      assert (PyList_Size(__ERROR) == 3);
      PyObject * err_type = PyList_GetItem(__ERROR, 0); //stolen ref
      PyObject * err_msg = PyList_GetItem(__ERROR, 1); //stolen ref
      PyObject * err_trace = PyList_GetItem(__ERROR, 2); //stolen ref
      PyList_SET_ITEM(__ERROR, 0, Py_None); Py_INCREF(Py_None); //clobbers old ref
      PyList_SET_ITEM(__ERROR, 1, Py_None); Py_INCREF(Py_None); //clobbers old ref
      PyList_SET_ITEM(__ERROR, 2, Py_None); Py_INCREF(Py_None); //clobbers old ref

      assert(!PyErr_Occurred()); // because CLinker hid the exception in __ERROR aka data
      PyErr_Restore(err_type, err_msg, err_trace); //steals refs to args
    }
  if (err) set_position_of_error(self, node_idx);
  return err;
}
static
int lazy_rec_eval(CLazyLinker * self, Py_ssize_t var_idx, PyObject*one, PyObject*zero)
{
  PyObject *rval = NULL;
  int verbose = 0;
  int err = 0;

  if (verbose) fprintf(stderr, "lazy_rec computing %i\n", (int)var_idx);

  if (self->var_computed[var_idx] || !self->var_has_owner[var_idx])
    return 0;

  Py_ssize_t owner_idx = self->var_owner[var_idx];

  // STEP 1: compute the pre-requirements of the node
  // Includes input nodes for non-lazy ops.
  for (int i = 0; i < self->node_n_prereqs[owner_idx]; ++i)
    {
      Py_ssize_t prereq_idx = self->node_prereqs[owner_idx][i];
      if (!self->var_computed[prereq_idx])
        {
          err = lazy_rec_eval(self, prereq_idx, one, zero);
          if (err) return err;
        }
      assert (self->var_computed[prereq_idx]);
    }

  // STEP 2: compute the node itself
  if (self->is_lazy[owner_idx])
    {
      // update the compute_map cells corresponding to the inputs of this thunk
      for (int i = 0; i < self->node_n_inputs[owner_idx]; ++i)
        {
          int in_idx = self->node_inputs[owner_idx][i];
          if (self->var_computed[in_idx])
            {
              Py_INCREF(one);
              err = PyList_SetItem(self->var_computed_cells[in_idx], 0, one);
            }
          else
            {
              Py_INCREF(zero);
              err = PyList_SetItem(self->var_computed_cells[in_idx], 0, zero);
            }
          if (err) goto fail;
        }

      rval = pycall(self, owner_idx, verbose);
      // refcounting - rval is new ref
      //TODO: to prevent infinite loops
      // - consider check that a thunk does not ask for an input that is already computed
      if (rval == NULL)
        {
          assert (PyErr_Occurred());
          err = 1;
          goto fail;
        }

      //update the computed-ness of any output cells
      for (int i = 0; i < self->node_n_outputs[owner_idx]; ++i)
        {
          int out_idx = self->node_outputs[owner_idx][i];
          PyObject * el_i = PyList_GetItem(self->var_computed_cells[out_idx], 0);
          Py_ssize_t N = PyNumber_AsSsize_t(el_i, PyExc_IndexError);
          if (PyErr_Occurred())
            {
              err = -1;
              goto pyfail;
            }
          assert (N==0 || N==1);
          self->var_computed[out_idx] = N;
        }
      if (!self->var_computed[var_idx])
        {
          /*
           * If self is not computed after the call, this means that some
           * inputs are needed.  Compute the ones on the returned list
           * and try to compute the current node again (with recursive call).
           * This allows a node to request more nodes more than once before
           * finally yielding a result.
           */
          if (!PyList_Check(rval))
            {
              //TODO: More helpful error to help find *which node* made this
              // bad thunk
              PyErr_SetString(PyExc_TypeError,
                              "lazy thunk should return a list");
              err = 1;
              goto pyfail;
            }

          if (!PyList_Size(rval))
            {
              PyErr_SetString(PyExc_ValueError,
                              "lazy thunk returned empty list without computing output");
              err = 1;
              goto pyfail;
            }

          for (int i = 0; i < PyList_Size(rval); ++i)
            {
              PyObject * el_i = PyList_GetItem(rval, i);
              Py_ssize_t N = PyNumber_AsSsize_t(el_i, PyExc_IndexError);
              if (PyErr_Occurred())
                {
                  err = 1;
                  goto pyfail;
                }
              assert (N <= self->node_n_inputs[owner_idx]);
              Py_ssize_t input_idx = self->node_inputs[owner_idx][N];
              err = lazy_rec_eval(self, input_idx, one, zero);
              if (err) goto pyfail;
            }

          Py_DECREF(rval);
          /*
           * We intentionally skip all the end-of-function processing
           * (mark outputs, GC) as it will be performed by the call
           * that actually manages to compute the result.
           */
          return lazy_rec_eval(self, var_idx, one, zero);
        }

      Py_DECREF(rval);
    }
  else //owner is not a lazy op. Ensure all intputs are evaluated.
    {
      // loop over inputs to owner
      // call lazy_rec_eval on each one that is not computed.
      // if there's an error, pass it up the stack
      for (int i = 0; i < self->node_n_inputs[owner_idx]; ++i)
        {
          Py_ssize_t input_idx = self->node_inputs[owner_idx][i];
          if (!self->var_computed[input_idx])
            {
              err = lazy_rec_eval(self, input_idx, one, zero);
              if (err) return err;
            }
          assert (self->var_computed[input_idx]);
        }

      // call the thunk for this owner.
      if (self->thunk_cptr_fn[owner_idx])
        {
          err = c_call(self, owner_idx, verbose);
          if (err) goto fail;
        }
      else
        {
          rval = pycall(self, owner_idx, verbose);
          //rval is new ref
          if (rval) //pycall returned normally (no exception)
            {
              if (rval == Py_None)
                {
                  Py_DECREF(rval); //ignore a return of None
                }
              else if (PyList_Check(rval))
                {
                  PyErr_SetString(PyExc_TypeError,
                                  "non-lazy thunk should return None, not list");
                  err = 1;
                  goto pyfail;
                }
              else // don't know what it returned, but it wasn't right.
                {
                  PyErr_SetObject(PyExc_TypeError, rval);
                  err = 1;
                  // We don't release rval since we put it in the error above
                  goto fail;
                }
            }
          else // pycall returned NULL (internal error)
            {
              err = 1;
              goto fail;
            }
        }
    }

  // loop over all outputs and mark them as computed
  for (int i = 0; i < self->node_n_outputs[owner_idx]; ++i)
    {
      self->var_computed[self->node_outputs[owner_idx][i]] = 1;
    }

  // Free vars that are not needed anymore
  if (self->allow_gc)
    {
      for (int i = 0; i < self->node_n_inputs[owner_idx]; ++i)
        {
          int cleanup = 1;
          Py_ssize_t i_idx = self->node_inputs[owner_idx][i];
          if (!self->var_has_owner[i_idx])
            continue;

          for (int j = 0; j < self->n_output_vars; ++j)
            {
              if (i_idx == self->output_vars[j])
                {
                  cleanup = 0;
                  break;
                }
            }
          if (!cleanup) continue;

          for (int j = 0; j < self->n_dependencies[i_idx]; ++j)
            {
              if (!self->var_computed[self->dependencies[i_idx][j]])
                {
                  cleanup = 0;
                  break;
                }
            }
          if (!cleanup) continue;

          Py_INCREF(Py_None);
          err = PyList_SetItem(self->var_value_cells[i_idx], 0, Py_None);
//See the Stack gc implementation for why we change it to 2 and not 0.
          self->var_computed[i_idx] = 2;
          if (err) goto fail;
        }
    }

  return 0;
 pyfail:
  Py_DECREF(rval);
 fail:
  set_position_of_error(self, owner_idx);
  return err;
}

static PyObject *
CLazyLinker_call(PyObject *_self, PyObject *args, PyObject *kwds)
{
  CLazyLinker * self = (CLazyLinker*)_self;
  static char *kwlist[] = {
    (char *)"time_thunks",
    (char *)"n_calls",
    (char *)"output_subset",
    NULL};
  int n_calls=1;
  PyObject *output_subset_ptr = NULL;
  if (! PyArg_ParseTupleAndKeywords(args, kwds, "|iiO", kwlist,
                                    &self->do_timing,
                                    &n_calls,
                                    &output_subset_ptr))
    return NULL;

  int err = 0;
  // parse an output_subset list
  // it is stored as a bool list of length n_output_vars: calculate a var or not
  char *output_subset = NULL;
  int output_subset_size = -1;
  if (output_subset_ptr != NULL)
    {
      if (! PyList_Check(output_subset_ptr))
        {
          err = 1;
          PyErr_SetString(PyExc_RuntimeError, "Output_subset is not a list");
        }
      else
        {
          output_subset_size = PyList_Size(output_subset_ptr);
          output_subset = (char*)calloc(self->n_output_vars, sizeof(char));
          for (int it = 0; it < output_subset_size; ++it)
            {
              PyObject *elem = PyList_GetItem(output_subset_ptr, it);
              if (! PyInt_Check(elem))
                {
                  err = 1;
                  PyErr_SetString(PyExc_RuntimeError, "Some elements of output_subset list are not int");
                }
              output_subset[PyInt_AsLong(elem)] = 1;
            }
        }
    }

  self->position_of_error = -1;
  // create constants used to fill the var_compute_cells
  PyObject * one = PyInt_FromLong(1);
  PyObject * zero = PyInt_FromLong(0);

  // pre-allocate our return value
  Py_INCREF(Py_None);
  PyObject * rval = Py_None;
  //clear storage of pre_call_clear elements
  for (int call_i = 0; call_i < n_calls && (!err); ++call_i)
    {
      Py_ssize_t n_pre_call_clear = PyList_Size(self->pre_call_clear);
      assert(PyList_Check(self->pre_call_clear));
      for (int i = 0; i < n_pre_call_clear; ++i)
        {
          PyObject * el_i = PyList_GetItem(self->pre_call_clear, i);
          Py_INCREF(Py_None);
          PyList_SetItem(el_i, 0, Py_None);
        }
      //clear the computed flag out of all non-input vars
      for (int i = 0; i < self->n_vars; ++i)
        {
          self->var_computed[i] = !self->var_has_owner[i];
          if (self->var_computed[i])
            {
              Py_INCREF(one);
              PyList_SetItem(self->var_computed_cells[i], 0, one);
            }
          else
            {
              Py_INCREF(zero);
              PyList_SetItem(self->var_computed_cells[i], 0, zero);
            }
        }

      int first_updated = self->n_output_vars - self->n_updates;
      for (int i = 0; i < self->n_output_vars && (!err); ++i)
        {
          if (i >= first_updated || output_subset == NULL || output_subset[i] == 1)
            {
              err = lazy_rec_eval(self, self->output_vars[i], one, zero);
            }
        }

      if (!err)
        {
          // save references to outputs prior to updating storage containers
          assert (self->n_output_vars >= self->n_updates);
          Py_DECREF(rval);
          rval = PyList_New(self->n_output_vars);
          for (int i = 0; i < (self->n_output_vars); ++i)
            {
              Py_ssize_t src = self->output_vars[i];
              PyObject * item = PyList_GetItem(self->var_value_cells[src], 0);
              if ((output_subset == NULL || output_subset[i]) &&
                  self->var_computed[src] != 1)
                {
                  err = 1;
                  PyErr_Format(PyExc_AssertionError,
                               "The compute map of output %d should contain "
                               "1 at the end of execution, not %d.",
                               i, self->var_computed[src]);
                  break;
                }
              Py_INCREF(item);
              PyList_SetItem(rval, i, item);
            }
        }

      if (!err)
        {
          // Update the inputs that have an update rule
          for (int i = 0; i < self->n_updates; ++i)
            {
              PyObject* tmp = PyList_GetItem(rval, self->n_output_vars - self->n_updates + i);
              Py_INCREF(tmp);
              Py_ssize_t dst = self->update_storage[i];
              PyList_SetItem(self->var_value_cells[dst], 0, tmp);
            }
        }
    }

  /*
    Clear everything that is left and not an output. This is needed
    for lazy evaluation since the current GC algo is too conservative
    with lazy graphs.
  */
  if (self->allow_gc && !err)
    {
      for (Py_ssize_t i = 0; i < self->n_vars; ++i)
        {
          int do_cleanup = 1;
          if (!self->var_has_owner[i] || !self->var_computed[i])
            continue;
          for (int j = 0; j < self->n_output_vars; ++j)
            {
              if (i == self->output_vars[j])
                {
                  do_cleanup = 0;
                  break;
                }
            }
          if (!do_cleanup)
            continue;
          Py_INCREF(Py_None);
          PyList_SetItem(self->var_value_cells[i], 0, Py_None);
        }
    }
  if (output_subset != NULL)
    free(output_subset);

  Py_DECREF(one);
  Py_DECREF(zero);
  if (err)
    {
      Py_DECREF(rval);
      return NULL;
    }
  return rval;
}

#if 0
static PyMethodDef CLazyLinker_methods[] = {
    {
      //"name", (PyCFunction)CLazyLinker_accept, METH_VARARGS, "Return the name, combining the first and last name"
    },
    {NULL}  /* Sentinel */
};
#endif


static PyObject *
CLazyLinker_get_allow_gc(CLazyLinker *self, void *closure)
{
    return PyBool_FromLong(self->allow_gc);
}

static int
CLazyLinker_set_allow_gc(CLazyLinker *self, PyObject *value, void *closure)
{
  if(!PyBool_Check(value))
    return -1;

  if (value == Py_True)
    self->allow_gc = true;
  else
    self->allow_gc = false;
  return 0;
}

static PyGetSetDef CLazyLinker_getset[] = {
  {(char*)"allow_gc",
   (getter)CLazyLinker_get_allow_gc,
   (setter)CLazyLinker_set_allow_gc,
   (char*)"do this function support allow_gc",
   NULL},
  {NULL, NULL, NULL, NULL}  /* Sentinel */
};
static PyMemberDef CLazyLinker_members[] = {
    {(char*)"nodes", T_OBJECT_EX, offsetof(CLazyLinker, nodes), 0,
     (char*)"list of nodes"},
    {(char*)"thunks", T_OBJECT_EX, offsetof(CLazyLinker, thunks), 0,
     (char*)"list of thunks in program"},
    {(char*)"call_counts", T_OBJECT_EX, offsetof(CLazyLinker, call_counts), 0,
     (char*)"number of calls of each thunk"},
    {(char*)"call_times", T_OBJECT_EX, offsetof(CLazyLinker, call_times), 0,
     (char*)"total runtime in each thunk"},
    {(char*)"position_of_error", T_INT, offsetof(CLazyLinker, position_of_error), 0,
     (char*)"position of failed thunk"},
    {(char*)"time_thunks", T_INT, offsetof(CLazyLinker, do_timing), 0,
     (char*)"bool: nonzero means call will time thunks"},
    {(char*)"need_update_inputs", T_INT, offsetof(CLazyLinker, need_update_inputs), 0,
     (char*)"bool: nonzero means Function.__call__ must implement update mechanism"},
    {NULL}  /* Sentinel */
};

static PyTypeObject lazylinker_ext_CLazyLinkerType = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "lazylinker_ext.CLazyLinker",             /*tp_name*/
    sizeof(CLazyLinker), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    CLazyLinker_dealloc,       /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    CLazyLinker_call,          /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT|Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    "CLazyLinker object",      /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,//CLazyLinker_methods,       /* tp_methods */
    CLazyLinker_members,       /* tp_members */
    CLazyLinker_getset,        /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)CLazyLinker_init,/* tp_init */
    0,                         /* tp_alloc */
    CLazyLinker_new,           /* tp_new */
};

static PyObject * get_version(PyObject *dummy, PyObject *args)
{
  PyObject *result = PyFloat_FromDouble(0.211);
  return result;
}

static PyMethodDef lazylinker_ext_methods[] = {
  {"get_version",  get_version, METH_VARARGS, "Get extension version."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "lazylinker_ext",
        NULL,
        -1,
        lazylinker_ext_methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif
#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC
PyInit_lazylinker_ext(void) {
#else
#define RETVAL
PyMODINIT_FUNC
initlazylinker_ext(void) 
{
#endif
    PyObject* m;

    lazylinker_ext_CLazyLinkerType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&lazylinker_ext_CLazyLinkerType) < 0)
        return RETVAL;
#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("lazylinker_ext", lazylinker_ext_methods,
                       "Example module that creates an extension type.");
#endif
    Py_INCREF(&lazylinker_ext_CLazyLinkerType);
    PyModule_AddObject(m, "CLazyLinker", (PyObject *)&lazylinker_ext_CLazyLinkerType);

    return RETVAL;
}
