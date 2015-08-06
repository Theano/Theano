#include <Python.h>
#include <cuda.h>

#define NAME        "cuda_devquery"
#define DOCSTRING   "A module for early CUDA device enumeration."

PyObject *
GetDeviceCount( PyObject* _unused, PyObject* dummy )
{
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount( &deviceCount );
    if( cudaSuccess != error_id ) {
        return PyErr_Format(
          PyExc_EnvironmentError,
          "Unable to get the number of GPUs available: %s",
          cudaGetErrorString( error_id ) );
    }

    return PyLong_FromLong( deviceCount );
}

PyObject * GetDeviceCapability(PyObject* _unused, PyObject* args)
{
  int dev_id = -1;
  if ( !PyArg_ParseTuple( args, "i", &dev_id ) )
    return PyErr_Format( PyExc_EnvironmentError, "Bad device number" );

  cudaDeviceProp deviceProp;
  cudaError_t error_id = cudaGetDeviceProperties( &deviceProp, dev_id );
  if( cudaSuccess != error_id ) {
      return PyErr_Format(
        PyExc_EnvironmentError,
        "Unable to query device %d: %s",
        dev_id, cudaGetErrorString( error_id ) );
  }

  return PyTuple_Pack( 2,
    PyLong_FromLong( deviceProp.major ),
    PyLong_FromLong( deviceProp.minor ) );
}

static PyMethodDef module_methods[ ] = {
    { "cuda_device_count", GetDeviceCount, METH_NOARGS,
      "Return the number of GPU devices available." },
    { "cuda_device_capability", GetDeviceCapability, METH_VARARGS,
      "Return GPU device capcbility in a (major, minor) tuple." },
    { NULL, NULL, NULL, NULL }
};

#if PY_MAJOR_VERSION == 3

static struct PyModuleDef cuda_devquery_moduledef =
    { PyModuleDef_HEAD_INIT, NAME, DOCSTRING, 0, module_methods };

PyMODINIT_FUNC PyInit_cuda_devquery( void )
    { return PyModule_Create( &cuda_devquery_moduledef ); }

#else

PyMODINIT_FUNC initcuda_devquery( void )
    { Py_InitModule3( NAME, module_methods, DOCSTRING ); }

#endif

