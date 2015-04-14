#ifndef THEANO_MOD_HELPER
#define THEANO_MOD_HELPER

#ifndef _WIN32
#define MOD_PUBLIC __attribute__((visibility ("default")))
#else
#define MOD_PUBLIC
#endif

#define THEANO_INIT_FUNC PyMODINIT_FUNC MOD_PUBLIC

#endif
