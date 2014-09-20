#ifndef THEANO_LAZYLINKER_C_H
#define THEANO_LAZYLINKER_C_H

#include <sys/time.h>
#include <Python.h>

double pytime(const struct timeval * tv);

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

PyObject *CLazyLinker_call(PyObject *_self, PyObject *args, PyObject *kwds);

#endif
