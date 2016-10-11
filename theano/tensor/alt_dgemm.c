/**
C Implementation of dgemm_ based on NumPy
Used instead of blas when Theano config flag blas.ldflags is empty.
**/
void alt_double_scalar_matrix_product_in_place(double scalar, double* matrix, int size_to_compute) {
	for(int i = 0; i < size_to_compute; ++i) {
		matrix[i] *= scalar;
	}
}
void alt_double_matrix_sum_in_place(double* A, double* B, double* out, int size_to_compute) {
	for(int i = 0; i < size_to_compute; ++i) {
		out[i] = A[i] + B[i];
	}
}
/* dgemm
 * NB: See sgemm_ (in alt_sgemm.c) for same assumptions.
 * */
void dgemm_(char* TRANSA, char* TRANSB, 
			const int* M, const int* N, const int* K,
			const double* ALPHA,  double* A, const int* LDA, 
			 double* B, const int* LDB, const double* BETA, 
			double* C, const int* LDC) {
	if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
		return;
	if(C == NULL)
		return;
	int ka, kb;
	if(*TRANSA == 'N' || *TRANSA == 'n')
		ka = *K;
	else
		ka = *M;
	if(*TRANSB == 'N' || *TRANSB == 'n')
		kb = *N;
	else
		kb = *K;
	npy_intp dims_A[2] = {*LDA, ka};
	npy_intp dims_B[2] = {*LDB, kb};
	PyObject* matrix_A = PyArray_SimpleNewFromData(2, dims_A, NPY_FLOAT64, A);
	PyObject* matrix_B = PyArray_SimpleNewFromData(2, dims_B, NPY_FLOAT64, B);
	PyObject* op_A = alt_op(TRANSA, (PyArrayObject*)matrix_A);
	PyObject* op_B = alt_op(TRANSB, (PyArrayObject*)matrix_B);
	if(*BETA == 0) {
		npy_intp dims_C[2] = {*LDC, *N};
		PyObject* matrix_C = PyArray_SimpleNewFromData(2, dims_C, NPY_FLOAT64, C);
		alt_matrix_matrix_product2(op_A, op_B, matrix_C);
		alt_double_scalar_matrix_product_in_place(*ALPHA, C, (*M) * (*N));
		Py_XDECREF(matrix_C);
	} else {
		PyArrayObject* op_A_times_op_B = (PyArrayObject*)alt_matrix_matrix_product(op_A, op_B);
		alt_double_scalar_matrix_product_in_place(*ALPHA, (double*)PyArray_DATA(op_A_times_op_B), (*M) * (*N));
		alt_double_scalar_matrix_product_in_place(*BETA, C, (*M) * (*N));
		alt_double_matrix_sum_in_place((double*)PyArray_DATA(op_A_times_op_B), C, C, (*M) * (*N));
		Py_XDECREF(op_A_times_op_B);
	}
	if(op_B != matrix_B) Py_XDECREF(op_B);
	if(op_A != matrix_A) Py_XDECREF(op_A);
	Py_XDECREF(matrix_B);
	Py_XDECREF(matrix_A);
}
