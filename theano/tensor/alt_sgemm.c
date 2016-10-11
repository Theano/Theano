/**
C Implementation of sgemm_ based on NumPy
Used instead of blas when Theano config flag blas.ldflags is empty.
**/
inline PyObject* alt_transpose(PyArrayObject* o) {
	return PyArray_Transpose(o, NULL);
}
inline PyObject* alt_matrix_matrix_product(PyObject* o1, PyObject* o2) {
	return PyArray_MatrixProduct(o1, o2);
}
inline PyObject* alt_matrix_matrix_product2(PyObject* o1, PyObject* o2, PyObject* out) {
	return PyArray_MatrixProduct2(o1, o2, (PyArrayObject*)out);
}
void alt_scalar_matrix_product_in_place(float scalar, float* matrix, int size_to_compute) {
	int i;
	for(i = 0; i < size_to_compute; ++i) {
		matrix[i] *= scalar;
	}
}
void alt_matrix_sum_in_place(float* A, float* B, float* out, int size_to_compute) {
	int i;
	for(i = 0; i < size_to_compute; ++i) {
		out[i] = A[i] + B[i];
	}
}
inline PyObject* alt_op(char* trans, PyArrayObject* matrix) {
	return (*trans == 'N' || *trans == 'n') ? (PyObject*)matrix : alt_transpose(matrix);
}
/* sgemm
 * We assume that none of these 13 pointers passed as arguments are null.
 * NB: We can optimize this function again (for example, when alpha == 0 and/or beta == 0).
 * */
void sgemm_(char* TRANSA, char* TRANSB, 
			const int* M, const int* N, const int* K,
			const float* ALPHA,  float* A, const int* LDA, 
			 float* B, const int* LDB, const float* BETA, 
			float* C, const int* LDC) {
	if(*M < 0 || *N < 0 || *K < 0 || *LDA < 0 || *LDB < 0 || *LDC < 0)
		return;
	if(C == NULL)
		return;
	/* Recall:
	A is a *LDA by *ka matrix.
	B is a *LDB by *kb matrix.
	C is a *LDC By *N  matrix.
	*/
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
	PyObject* matrix_A = PyArray_SimpleNewFromData(2, dims_A, NPY_FLOAT32, A);
	PyObject* matrix_B = PyArray_SimpleNewFromData(2, dims_B, NPY_FLOAT32, B);
	PyObject* op_A = alt_op(TRANSA, (PyArrayObject*)matrix_A);
	PyObject* op_B = alt_op(TRANSB, (PyArrayObject*)matrix_B);
	if(*BETA == 0) {
		npy_intp dims_C[2] = {*LDC, *N};
		PyObject* matrix_C = PyArray_SimpleNewFromData(2, dims_C, NPY_FLOAT32, C);
		alt_matrix_matrix_product2(op_A, op_B, matrix_C);
		alt_scalar_matrix_product_in_place(*ALPHA, C, (*M) * (*N));
		Py_XDECREF(matrix_C);
	} else {
		PyArrayObject* op_A_times_op_B = (PyArrayObject*)alt_matrix_matrix_product(op_A, op_B);
		alt_scalar_matrix_product_in_place(*ALPHA, (float*)PyArray_DATA(op_A_times_op_B), (*M) * (*N));
		alt_scalar_matrix_product_in_place(*BETA, C, (*M) * (*N));
		alt_matrix_sum_in_place((float*)PyArray_DATA(op_A_times_op_B), C, C, (*M) * (*N));
		Py_XDECREF(op_A_times_op_B);
	}
	if(op_B != matrix_B) Py_XDECREF(op_B);
	if(op_A != matrix_A) Py_XDECREF(op_A);
	Py_XDECREF(matrix_B);
	Py_XDECREF(matrix_A);
}
