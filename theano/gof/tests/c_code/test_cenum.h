#ifndef THEANO_TEST_CENUM
#define THEANO_TEST_CENUM

#define SIZE_INT 0
#define SIZE_FLOAT 1
#define SIZE_LONG_LONG 2

/* NB:
For this specific test, we can not directly use macros with corresponding values (e.g. `SIZEOF_INT == sizeof(int)`)
because different types may have same size (e.g. sizeof(int) and sizeof(float) will be 4 on most machines),
leading to compilation errors in debug mode (inside a switch loop, SIZE_INT and SIZE_FLOAT would have the same
value, so that it is impossible to distinguish them, for e.g. to print macro name).
*/

static size_t type_sizes[] = {sizeof(int), sizeof(float), sizeof(long long)};

#endif
