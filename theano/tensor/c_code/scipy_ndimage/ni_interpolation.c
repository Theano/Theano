//
// This file was part of SciPi (scipy/ndimage/src/ni_interpolation.c),
// copied from SciPi commit 7c28f602c6bff1548ffaa4afa7597606cdd7cf9e
// on 15 September 2017.
//
// There are some modifications to make it work in Theano.
//  - added some casts to malloc statements
//  - added NI_SplineFilter1DGrad
//  - added 'reverse' parameter to NI_ZoomShift to compute the gradient
//

/* Copyright (C) 2003-2005 Peter J. Verveer
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// disabled for Theano
// #include "ni_support.h"
// #include "ni_interpolation.h"
// #include <stdlib.h>
// #include <math.h>

/* calculate the B-spline interpolation coefficients for given x: */
static void
spline_coefficients(double x, int order, double *result)
{
    int hh;
    double y, start;

    if (order & 1) {
        start = (int)floor(x) - order / 2;
    } else {
        start = (int)floor(x + 0.5) - order / 2;
    }

    for(hh = 0; hh <= order; hh++)  {
        y = fabs(start - x + hh);

        switch(order) {
        case 1:
            result[hh] = y > 1.0 ? 0.0 : 1.0 - y;
            break;
        case 2:
            if (y < 0.5) {
                result[hh] = 0.75 - y * y;
            } else if (y < 1.5) {
                y = 1.5 - y;
                result[hh] = 0.5 * y * y;
            } else {
                result[hh] = 0.0;
            }
            break;
        case 3:
            if (y < 1.0) {
                result[hh] =
                    (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
            } else if (y < 2.0) {
                y = 2.0 - y;
                result[hh] = y * y * y / 6.0;
            } else {
                result[hh] = 0.0;
            }
            break;
        case 4:
            if (y < 0.5) {
                y *= y;
                result[hh] = y * (y * 0.25 - 0.625) + 115.0 / 192.0;
            } else if (y < 1.5) {
                result[hh] = y * (y * (y * (5.0 / 6.0 - y / 6.0) - 1.25) +
                                                    5.0 / 24.0) + 55.0 / 96.0;
            } else if (y < 2.5) {
                y -= 2.5;
                y *= y;
                result[hh] = y * y / 24.0;
            } else {
                result[hh] = 0.0;
            }
            break;
        case 5:
            if (y < 1.0) {
                double f = y * y;
                result[hh] =
                    f * (f * (0.25 - y / 12.0) - 0.5) + 0.55;
            } else if (y < 2.0) {
                result[hh] = y * (y * (y * (y * (y / 24.0 - 0.375)
                                                                        + 1.25) -  1.75) + 0.625) + 0.425;
            } else if (y < 3.0) {
                double f = 3.0 - y;
                y = f * f;
                result[hh] = f * y * y / 120.0;
            } else {
                result[hh] = 0.0;
            }
            break;
        }
    }
}

/* map a coordinate outside the borders, according to the requested
     boundary condition: */
static double
map_coordinate(double in, npy_intp len, int mode)
{
    if (in < 0) {
        switch (mode) {
        case NI_EXTEND_MIRROR:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz2 = 2 * len - 2;
                in = sz2 * (npy_intp)(-in / sz2) + in;
                in = in <= 1 - len ? in + sz2 : -in;
            }
            break;
        case NI_EXTEND_REFLECT:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz2 = 2 * len;
                if (in < -sz2)
                    in = sz2 * (npy_intp)(-in / sz2) + in;
                in = in < -len ? in + sz2 : -in - 1;
            }
            break;
        case NI_EXTEND_WRAP:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz = len - 1;
                // Integer division of -in/sz gives (-in mod sz)
                // Note that 'in' is negative
                in += sz * ((npy_intp)(-in / sz) + 1);
            }
            break;
        case NI_EXTEND_NEAREST:
            in = 0;
            break;
        case NI_EXTEND_CONSTANT:
            in = -1;
            break;
        }
    } else if (in > len-1) {
        switch (mode) {
        case NI_EXTEND_MIRROR:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz2 = 2 * len - 2;
                in -= sz2 * (npy_intp)(in / sz2);
                if (in >= len)
                    in = sz2 - in;
            }
            break;
        case NI_EXTEND_REFLECT:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz2 = 2 * len;
                in -= sz2 * (npy_intp)(in / sz2);
                if (in >= len)
                    in = sz2 - in - 1;
            }
            break;
        case NI_EXTEND_WRAP:
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz = len - 1;
                in -= sz * (npy_intp)(in / sz);
            }
            break;
        case NI_EXTEND_NEAREST:
            in = len - 1;
            break;
        case NI_EXTEND_CONSTANT:
            in = -1;
            break;
        }
    }

    return in;
}

#define BUFFER_SIZE 256000
#define TOLERANCE 1e-15

/* one-dimensional spline filter: */
int NI_SplineFilter1D(PyArrayObject *input, int order, int axis,
                                            PyArrayObject *output)
{
    int hh, npoles = 0, more;
    npy_intp kk, ll, lines, len;
    double *buffer = NULL, weight, pole[2];
    NI_LineBuffer iline_buffer, oline_buffer;
    NPY_BEGIN_THREADS_DEF;

    len = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;
    if (len < 1)
        goto exit;

    /* these are used in the spline filter calculation below: */
    switch (order) {
    case 2:
        npoles = 1;
        pole[0] = sqrt(8.0) - 3.0;
        break;
    case 3:
        npoles = 1;
        pole[0] = sqrt(3.0) - 2.0;
        break;
    case 4:
        npoles = 2;
        pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
        pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
        break;
    case 5:
        npoles = 2;
        pole[0] = sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5;
        pole[1] = sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5;
        break;
    default:
        break;
    }

    weight = 1.0;
    for(hh = 0; hh < npoles; hh++)
        weight *= (1.0 - pole[hh]) * (1.0 - 1.0 / pole[hh]);

    /* allocate an initialize the line buffer, only a single one is used,
         because the calculation is in-place: */
    lines = -1;
    if (!NI_AllocateLineBuffer(input, axis, 0, 0, &lines, BUFFER_SIZE,
                                                         &buffer))
        goto exit;
    if (!NI_InitLineBuffer(input, axis, 0, 0, lines, buffer,
                                                 NI_EXTEND_DEFAULT, 0.0, &iline_buffer))
        goto exit;
    if (!NI_InitLineBuffer(output, axis, 0, 0, lines, buffer,
                                                 NI_EXTEND_DEFAULT, 0.0, &oline_buffer))
        goto exit;

    NPY_BEGIN_THREADS;

    /* iterate over all the array lines: */
    do {
        /* copy lines from array to buffer: */
        if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {
            goto exit;
        }
        /* iterate over the lines in the buffer: */
        for(kk = 0; kk < lines; kk++) {
            /* get line: */
            double *ln = NI_GET_LINE(iline_buffer, kk);
            /* spline filter: */
            if (len > 1) {
                for(ll = 0; ll < len; ll++)
                    ln[ll] *= weight;
                for(hh = 0; hh < npoles; hh++) {
                    double p = pole[hh];
                    int max = (int)ceil(log(TOLERANCE) / log(fabs(p)));
                    if (max < len) {
                        double zn = p;
                        double sum = ln[0];
                        for(ll = 1; ll < max; ll++) {
                            sum += zn * ln[ll];
                            zn *= p;
                        }
                        ln[0] = sum;
                    } else {
                        double zn = p;
                        double iz = 1.0 / p;
                        double z2n = pow(p, (double)(len - 1));
                        double sum = ln[0] + z2n * ln[len - 1];
                        z2n *= z2n * iz;
                        for(ll = 1; ll <= len - 2; ll++) {
                            sum += (zn + z2n) * ln[ll];
                            zn *= p;
                            z2n *= iz;
                        }
                        ln[0] = sum / (1.0 - zn * zn);
                    }
                    for(ll = 1; ll < len; ll++)
                        ln[ll] += p * ln[ll - 1];
                    ln[len-1] = (p / (p * p - 1.0)) * (ln[len-1] + p * ln[len-2]);
                    for(ll = len - 2; ll >= 0; ll--)
                        ln[ll] = p * (ln[ll + 1] - ln[ll]);
                }
            }
        }
        /* copy lines from buffer to array: */
        if (!NI_LineBufferToArray(&oline_buffer)) {
            goto exit;
        }
    } while(more);

 exit:
    NPY_END_THREADS;
    free(buffer);
    return PyErr_Occurred() ? 0 : 1;
}

// added for Theano
/* gradient of one-dimensional spline filter: */
int NI_SplineFilter1DGrad(PyArrayObject *input, int order, int axis,
                                                PyArrayObject *output)
{
    int hh, npoles = 0, more;
    npy_intp kk, ll, lines, len;
    double *buffer = NULL, weight, pole[2];
    NI_LineBuffer iline_buffer, oline_buffer;
    NPY_BEGIN_THREADS_DEF;

    len = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;
    if (len < 1)
        goto exit;

    /* these are used in the spline filter calculation below: */
    switch (order) {
    case 2:
        npoles = 1;
        pole[0] = sqrt(8.0) - 3.0;
        break;
    case 3:
        npoles = 1;
        pole[0] = sqrt(3.0) - 2.0;
        break;
    case 4:
        npoles = 2;
        pole[0] = sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0;
        pole[1] = sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0;
        break;
    case 5:
        npoles = 2;
        pole[0] = sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5;
        pole[1] = sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5;
        break;
    default:
        break;
    }

    weight = 1.0;
    for(hh = 0; hh < npoles; hh++)
        weight *= (1.0 - pole[hh]) * (1.0 - 1.0 / pole[hh]);

    /* allocate an initialize the line buffer, only a single one is used,
         because the calculation is in-place: */
    lines = -1;
    if (!NI_AllocateLineBuffer(input, axis, 0, 0, &lines, BUFFER_SIZE,
                                                         &buffer))
        goto exit;
    if (!NI_InitLineBuffer(input, axis, 0, 0, lines, buffer,
                                                 NI_EXTEND_DEFAULT, 0.0, &iline_buffer))
        goto exit;
    if (!NI_InitLineBuffer(output, axis, 0, 0, lines, buffer,
                                                 NI_EXTEND_DEFAULT, 0.0, &oline_buffer))
        goto exit;

    NPY_BEGIN_THREADS;

    /* iterate over all the array lines: */
    do {
        /* copy lines from array to buffer: */
        if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {
            goto exit;
        }
        /* iterate over the lines in the buffer: */
        for(kk = 0; kk < lines; kk++) {
            /* get line: */
            double *ln = NI_GET_LINE(iline_buffer, kk);
            /* spline filter: */
            if (len > 1) {
                for(hh = 0; hh < npoles; hh++) {
                    double p = pole[hh];
                    int max = (int)ceil(log(TOLERANCE) / log(fabs(p)));

                    double sum = p * ln[0];
                    ln[0] = -p * ln[0];
                    for(ll = 1; ll < len - 1; ll++) {
                        sum = p * (sum + ln[ll]);
                        ln[ll] = p * (ln[ll - 1] - ln[ll]);
                    }
                    sum = (p / (p * p - 1.0)) * (sum + ln[len - 1]);
                    ln[len - 2] += p * sum;
                    ln[len - 1] = sum;

                    for(ll = len - 2; ll >= 0; ll--)
                        ln[ll] += p * ln[ll + 1];

                    if (max < len) {
                        double zn = p;
                        for(ll = 1; ll < len; ll++) {
                            ln[ll] += zn * ln[0];
                            zn *= p;
                        }
                    } else {
                        double zn = p;
                        double iz = 1.0 / p;
                        double z2n = pow(p, (double)(len - 1));
                        ln[0] = ln[0] / (1.0 - z2n * z2n);
                        ln[len - 1] += z2n * ln[0];
                        z2n *= z2n * iz;
                        for(ll = 1; ll <= len - 2; ll++) {
                            ln[ll] += (zn + z2n) * ln[0];
                            zn *= p;
                            z2n *= iz;
                        }
                    }
                }
                for(ll = 0; ll < len; ll++)
                    ln[ll] *= weight;
            }
        }
        /* copy lines from buffer to array: */
        if (!NI_LineBufferToArray(&oline_buffer)) {
            goto exit;
        }
    } while(more);

 exit:
    NPY_END_THREADS;
    free(buffer);
    return PyErr_Occurred() ? 0 : 1;
}

/* copy row of coordinate array from location at _p to _coor */
#define CASE_MAP_COORDINATES(_TYPE, _type, _p, _coor, _rank, _stride) \
case _TYPE:                                                           \
{                                                                     \
    npy_intp _hh;                                                     \
    for (_hh = 0; _hh < _rank; ++_hh) {                               \
        _coor[_hh] = *(_type *)_p;                                    \
        _p += _stride;                                                \
    }                                                                 \
}                                                                     \
break

#define CASE_INTERP_COEFF(_TYPE, _type, _coeff, _pi, _idx) \
case _TYPE:                                                \
    _coeff = *(_type *)(_pi + _idx);                       \
    break

#define CASE_INTERP_OUT(_TYPE, _type, _po, _t) \
case _TYPE:                                    \
    *(_type *)_po = (_type)_t;                 \
    break

#define CASE_INTERP_OUT_UINT(_TYPE, _type, _po, _t)  \
case NPY_##_TYPE:                                    \
    _t = _t > 0 ? _t + 0.5 : 0;                      \
    _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t; \
    _t = _t < 0 ? 0 : t;                             \
    *(_type *)_po = (_type)_t;                       \
    break

#define CASE_INTERP_OUT_INT(_TYPE, _type, _po, _t)   \
case NPY_##_TYPE:                                    \
    _t = _t > 0 ? _t + 0.5 : _t - 0.5;               \
    _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t; \
    _t = _t < NPY_MIN_##_TYPE ? NPY_MIN_##_TYPE : t; \
    *(_type *)_po = (_type)_t;                       \
    break

// added for Theano
#define CASE_INTERP_INCR(_TYPE, _type, _po, _t) \
case _TYPE:                                     \
    *(_type *)_po += (_type)_t;                 \
    break

// added for Theano
#define CASE_INTERP_INCR_UINT(_TYPE, _type, _po, _t)  \
case NPY_##_TYPE:                                     \
    _t += *(_type *)_po;                              \
    _t = _t > 0 ? _t + 0.5 : 0;                       \
    _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t;  \
    _t = _t < 0 ? 0 : t;                              \
    *(_type *)_po = (_type)_t;                        \
    break

// added for Theano
#define CASE_INTERP_INCR_INT(_TYPE, _type, _po, _t)   \
case NPY_##_TYPE:                                     \
    _t += *(_type *)_po;                              \
    _t = _t > 0 ? _t + 0.5 : _t - 0.5;                \
    _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t;  \
    _t = _t < NPY_MIN_##_TYPE ? NPY_MIN_##_TYPE : t;  \
    *(_type *)_po = (_type)_t;                        \
    break

// added for Theano
#define CASE_INTERP_GRAD(_TYPE, _type, _grad, _po)  \
case NPY_##_TYPE:                                   \
    _grad = *(_type *)(_po);                        \
    break

int
NI_GeometricTransform(PyArrayObject *input, int (*map)(npy_intp*, double*,
                int, int, void*), void* map_data, PyArrayObject* matrix_ar,
                PyArrayObject* shift_ar, PyArrayObject *coordinates,
                PyArrayObject *output, int order, int mode, double cval)
{
    char *po, *pi, *pc = NULL;
    npy_intp **edge_offsets = NULL, **data_offsets = NULL, filter_size;
    npy_intp ftmp[NPY_MAXDIMS], *fcoordinates = NULL, *foffsets = NULL;
    npy_intp cstride = 0, kk, hh, ll, jj;
    npy_intp size;
    double **splvals = NULL, icoor[NPY_MAXDIMS];
    npy_intp idimensions[NPY_MAXDIMS], istrides[NPY_MAXDIMS];
    NI_Iterator io, ic;
    npy_double *matrix = matrix_ar ? (npy_double*)PyArray_DATA(matrix_ar) : NULL;
    npy_double *shift = shift_ar ? (npy_double*)PyArray_DATA(shift_ar) : NULL;
    int irank = 0, orank;
    NPY_BEGIN_THREADS_DEF;

    NPY_BEGIN_THREADS;

    for(kk = 0; kk < PyArray_NDIM(input); kk++) {
        idimensions[kk] = PyArray_DIM(input, kk);
        istrides[kk] = PyArray_STRIDE(input, kk);
    }
    irank = PyArray_NDIM(input);
    orank = PyArray_NDIM(output);

    /* if the mapping is from array coordinates: */
    if (coordinates) {
        /* initialze a line iterator along the first axis: */
        if (!NI_InitPointIterator(coordinates, &ic))
            goto exit;
        cstride = ic.strides[0];
        if (!NI_LineIterator(&ic, 0))
            goto exit;
        pc = (char *)(PyArray_DATA(coordinates));
    }

    /* offsets used at the borders: */
    edge_offsets = (npy_intp**)malloc(irank * sizeof(npy_intp*));
    data_offsets = (npy_intp**)malloc(irank * sizeof(npy_intp*));
    if (NPY_UNLIKELY(!edge_offsets || !data_offsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank; jj++)
        data_offsets[jj] = NULL;
    for(jj = 0; jj < irank; jj++) {
        data_offsets[jj] = (npy_intp*)malloc((order + 1) * sizeof(npy_intp));
        if (NPY_UNLIKELY(!data_offsets[jj])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }
    /* will hold the spline coefficients: */
    splvals = (double**)malloc(irank * sizeof(double*));
    if (NPY_UNLIKELY(!splvals)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank; jj++)
        splvals[jj] = NULL;
    for(jj = 0; jj < irank; jj++) {
        splvals[jj] = (double*)malloc((order + 1) * sizeof(double));
        if (NPY_UNLIKELY(!splvals[jj])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }

    filter_size = 1;
    for(jj = 0; jj < irank; jj++)
        filter_size *= order + 1;

    /* initialize output iterator: */
    if (!NI_InitPointIterator(output, &io))
        goto exit;

    /* get data pointers: */
    pi = (char *)PyArray_DATA(input);
    po = (char *)PyArray_DATA(output);

    /* make a table of all possible coordinates within the spline filter: */
    fcoordinates = (npy_intp*)malloc(irank * filter_size * sizeof(npy_intp));
    /* make a table of all offsets within the spline filter: */
    foffsets = (npy_intp*)malloc(filter_size * sizeof(npy_intp));
    if (NPY_UNLIKELY(!fcoordinates || !foffsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < irank; jj++)
        ftmp[jj] = 0;
    kk = 0;
    for(hh = 0; hh < filter_size; hh++) {
        for(jj = 0; jj < irank; jj++)
            fcoordinates[jj + hh * irank] = ftmp[jj];
        foffsets[hh] = kk;
        for(jj = irank - 1; jj >= 0; jj--) {
            if (ftmp[jj] < order) {
                ftmp[jj]++;
                kk += istrides[jj];
                break;
            } else {
                ftmp[jj] = 0;
                kk -= istrides[jj] * order;
            }
        }
    }

    size = PyArray_SIZE(output);
    for(kk = 0; kk < size; kk++) {
        double t = 0.0;
        int constant = 0, edge = 0;
        npy_intp offset = 0;
        if (map) {
            NPY_END_THREADS;
            /* call mappint functions: */
            if (!map(io.coordinates, icoor, orank, irank, map_data)) {
                if (!PyErr_Occurred())
                    PyErr_SetString(PyExc_RuntimeError,
                                                    "unknown error in mapping function");
                goto exit;
            }
            NPY_BEGIN_THREADS;
        } else if (matrix) {
            /* do an affine transformation: */
            npy_double *p = matrix;
            for(hh = 0; hh < irank; hh++) {
                icoor[hh] = 0.0;
                for(ll = 0; ll < orank; ll++)
                    icoor[hh] += io.coordinates[ll] * *p++;
                icoor[hh] += shift[hh];
            }
        } else if (coordinates) {
            /* mapping is from an coordinates array: */
            char *p = pc;
            switch (PyArray_TYPE(coordinates)) {
                CASE_MAP_COORDINATES(NPY_BOOL, npy_bool,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_UBYTE, npy_ubyte,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_USHORT, npy_ushort,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_UINT, npy_uint,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_ULONG, npy_ulong,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_ULONGLONG, npy_ulonglong,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_BYTE, npy_byte,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_SHORT, npy_short,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_INT, npy_int,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_LONG, npy_long,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_LONGLONG, npy_longlong,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_FLOAT, npy_float,
                                     p, icoor, irank, cstride);
                CASE_MAP_COORDINATES(NPY_DOUBLE, npy_double,
                                     p, icoor, irank, cstride);
            default:
                NPY_END_THREADS;
                PyErr_SetString(PyExc_RuntimeError,
                                "coordinate array data type not supported");
                goto exit;
            }
        }
        /* iterate over axes: */
        for(hh = 0; hh < irank; hh++) {
            /* if the input coordinate is outside the borders, map it: */
            double cc = map_coordinate(icoor[hh], idimensions[hh], mode);
            if (cc > -1.0) {
                /* find the filter location along this axis: */
                npy_intp start;
                if (order & 1) {
                    start = (npy_intp)floor(cc) - order / 2;
                } else {
                    start = (npy_intp)floor(cc + 0.5) - order / 2;
                }
                /* get the offset to the start of the filter: */
                offset += istrides[hh] * start;
                if (start < 0 || start + order >= idimensions[hh]) {
                    /* implement border mapping, if outside border: */
                    edge = 1;
                    edge_offsets[hh] = data_offsets[hh];
                    for(ll = 0; ll <= order; ll++) {
                        npy_intp idx = start + ll;
                        npy_intp len = idimensions[hh];
                        if (len <= 1) {
                            idx = 0;
                        } else {
                            npy_intp s2 = 2 * len - 2;
                            if (idx < 0) {
                                idx = s2 * (int)(-idx / s2) + idx;
                                idx = idx <= 1 - len ? idx + s2 : -idx;
                            } else if (idx >= len) {
                                idx -= s2 * (int)(idx / s2);
                                if (idx >= len)
                                    idx = s2 - idx;
                            }
                        }
                        /* calculate and store the offests at this edge: */
                        edge_offsets[hh][ll] = istrides[hh] * (idx - start);
                    }
                } else {
                    /* we are not at the border, use precalculated offsets: */
                    edge_offsets[hh] = NULL;
                }
                spline_coefficients(cc, order, splvals[hh]);
            } else {
                /* we use the constant border condition: */
                constant = 1;
                break;
            }
        }

        if (!constant) {
            npy_intp *ff = fcoordinates;
            const int type_num = PyArray_TYPE(input);
            t = 0.0;
            for(hh = 0; hh < filter_size; hh++) {
                double coeff = 0.0;
                npy_intp idx = 0;

                if (NPY_UNLIKELY(edge)) {
                    for(ll = 0; ll < irank; ll++) {
                        if (edge_offsets[ll])
                            idx += edge_offsets[ll][ff[ll]];
                        else
                            idx += ff[ll] * istrides[ll];
                    }
                } else {
                    idx = foffsets[hh];
                }
                idx += offset;
                switch (type_num) {
                    CASE_INTERP_COEFF(NPY_BOOL, npy_bool,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_UBYTE, npy_ubyte,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_USHORT, npy_ushort,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_UINT, npy_uint,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_ULONG, npy_ulong,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_ULONGLONG, npy_ulonglong,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_BYTE, npy_byte,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_SHORT, npy_short,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_INT, npy_int,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_LONG, npy_long,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_LONGLONG, npy_longlong,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_FLOAT, npy_float,
                                      coeff, pi, idx);
                    CASE_INTERP_COEFF(NPY_DOUBLE, npy_double,
                                      coeff, pi, idx);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError,
                                    "data type not supported");
                    goto exit;
                }
                /* calculate the interpolated value: */
                for(ll = 0; ll < irank; ll++)
                    if (order > 0)
                        coeff *= splvals[ll][ff[ll]];
                t += coeff;
                ff += irank;
            }
        } else {
            t = cval;
        }
        /* store output value: */
        switch (PyArray_TYPE(output)) {
            CASE_INTERP_OUT(NPY_BOOL, npy_bool, po, t);
            CASE_INTERP_OUT_UINT(UBYTE, npy_ubyte, po, t);
            CASE_INTERP_OUT_UINT(USHORT, npy_ushort, po, t);
            CASE_INTERP_OUT_UINT(UINT, npy_uint, po, t);
            CASE_INTERP_OUT_UINT(ULONG, npy_ulong, po, t);
            CASE_INTERP_OUT_UINT(ULONGLONG, npy_ulonglong, po, t);
            CASE_INTERP_OUT_INT(BYTE, npy_byte, po, t);
            CASE_INTERP_OUT_INT(SHORT, npy_short, po, t);
            CASE_INTERP_OUT_INT(INT, npy_int, po, t);
            CASE_INTERP_OUT_INT(LONG, npy_long, po, t);
            CASE_INTERP_OUT_INT(LONGLONG, npy_longlong, po, t);
            CASE_INTERP_OUT(NPY_FLOAT, npy_float, po, t);
            CASE_INTERP_OUT(NPY_DOUBLE, npy_double, po, t);
        default:
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
        }
        if (coordinates) {
            NI_ITERATOR_NEXT2(io, ic, po, pc);
        } else {
            NI_ITERATOR_NEXT(io, po);
        }
    }

 exit:
    NPY_END_THREADS;
    free(edge_offsets);
    if (data_offsets) {
        for(jj = 0; jj < irank; jj++)
            free(data_offsets[jj]);
        free(data_offsets);
    }
    if (splvals) {
        for(jj = 0; jj < irank; jj++)
            free(splvals[jj]);
        free(splvals);
    }
    free(foffsets);
    free(fcoordinates);
    return PyErr_Occurred() ? 0 : 1;
}

// modification for Theano:
// - if reverse is false, this function computes the forward case,
//   reading from input and writing to output (the default)
// - if reverse is true, this function will compute the gradient
//   by reading from output and writing to input
int NI_ZoomShift(PyArrayObject *input, PyArrayObject* zoom_ar,
                                 PyArrayObject* shift_ar, PyArrayObject *output,
                                 int order, int mode, double cval, int reverse)
{
    char *po, *pi;
    npy_intp **zeros = NULL, **offsets = NULL, ***edge_offsets = NULL;
    npy_intp ftmp[NPY_MAXDIMS], *fcoordinates = NULL, *foffsets = NULL;
    npy_intp jj, hh, kk, filter_size, odimensions[NPY_MAXDIMS];
    npy_intp idimensions[NPY_MAXDIMS], istrides[NPY_MAXDIMS];
    npy_intp size;
    double ***splvals = NULL;
    NI_Iterator io;
    npy_double *zooms = zoom_ar ? (npy_double*)PyArray_DATA(zoom_ar) : NULL;
    npy_double *shifts = shift_ar ? (npy_double*)PyArray_DATA(shift_ar) : NULL;
    int rank = 0;
    NPY_BEGIN_THREADS_DEF;

    NPY_BEGIN_THREADS;

    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
        idimensions[kk] = PyArray_DIM(input, kk);
        istrides[kk] = PyArray_STRIDE(input, kk);
        odimensions[kk] = PyArray_DIM(output, kk);
    }
    rank = PyArray_NDIM(input);

    /* if the mode is 'constant' we need some temps later: */
    if (mode == NI_EXTEND_CONSTANT) {
        zeros = (npy_intp**)malloc(rank * sizeof(npy_intp*));
        if (NPY_UNLIKELY(!zeros)) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
        for(jj = 0; jj < rank; jj++)
            zeros[jj] = NULL;
        for(jj = 0; jj < rank; jj++) {
            zeros[jj] = (npy_intp*)malloc(odimensions[jj] * sizeof(npy_intp));
            if (NPY_UNLIKELY(!zeros[jj])) {
                NPY_END_THREADS;
                PyErr_NoMemory();
                goto exit;
            }
        }
    }

    /* store offsets, along each axis: */
    offsets = (npy_intp**)malloc(rank * sizeof(npy_intp*));
    /* store spline coefficients, along each axis: */
    splvals = (double***)malloc(rank * sizeof(double**));
    /* store offsets at all edges: */
    edge_offsets = (npy_intp***)malloc(rank * sizeof(npy_intp**));
    if (NPY_UNLIKELY(!offsets || !splvals || !edge_offsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < rank; jj++) {
        offsets[jj] = NULL;
        splvals[jj] = NULL;
        edge_offsets[jj] = NULL;
    }
    for(jj = 0; jj < rank; jj++) {
        offsets[jj] = (npy_intp*)malloc(odimensions[jj] * sizeof(npy_intp));
        splvals[jj] = (double**)malloc(odimensions[jj] * sizeof(double*));
        edge_offsets[jj] = (npy_intp**)malloc(odimensions[jj] * sizeof(npy_intp*));
        if (NPY_UNLIKELY(!offsets[jj] || !splvals[jj] || !edge_offsets[jj])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
        for(hh = 0; hh < odimensions[jj]; hh++) {
            splvals[jj][hh] = NULL;
            edge_offsets[jj][hh] = NULL;
        }
    }

    /* precalculate offsets, and offsets at the edge: */
    for(jj = 0; jj < rank; jj++) {
        double shift = 0.0, zoom = 0.0;
        if (shifts)
            shift = shifts[jj];
        if (zooms)
            zoom = zooms[jj];
        for(kk = 0; kk < odimensions[jj]; kk++) {
            double cc = (double)kk;
            if (shifts)
                cc += shift;
            if (zooms)
                cc *= zoom;
            cc = map_coordinate(cc, idimensions[jj], mode);
            if (cc > -1.0) {
                npy_intp start;
                if (zeros && zeros[jj])
                    zeros[jj][kk] = 0;
                if (order & 1) {
                    start = (npy_intp)floor(cc) - order / 2;
                } else {
                    start = (npy_intp)floor(cc + 0.5) - order / 2;
                }
                offsets[jj][kk] = istrides[jj] * start;
                if (start < 0 || start + order >= idimensions[jj]) {
                    edge_offsets[jj][kk] = (npy_intp*)malloc((order + 1) * sizeof(npy_intp));
                    if (NPY_UNLIKELY(!edge_offsets[jj][kk])) {
                        NPY_END_THREADS;
                        PyErr_NoMemory();
                        goto exit;
                    }
                    for(hh = 0; hh <= order; hh++) {
                        npy_intp idx = start + hh;
                        npy_intp len = idimensions[jj];
                        if (len <= 1) {
                            idx = 0;
                        } else {
                            npy_intp s2 = 2 * len - 2;
                            if (idx < 0) {
                                idx = s2 * (npy_intp)(-idx / s2) + idx;
                                idx = idx <= 1 - len ? idx + s2 : -idx;
                            } else if (idx >= len) {
                                idx -= s2 * (npy_intp)(idx / s2);
                                if (idx >= len)
                                    idx = s2 - idx;
                            }
                        }
                        edge_offsets[jj][kk][hh] = istrides[jj] * (idx - start);
                    }
                }
                if (order > 0) {
                    splvals[jj][kk] = (double*)malloc((order + 1) * sizeof(double));
                    if (NPY_UNLIKELY(!splvals[jj][kk])) {
                        NPY_END_THREADS;
                        PyErr_NoMemory();
                        goto exit;
                    }
                    spline_coefficients(cc, order, splvals[jj][kk]);
                }
            } else {
                zeros[jj][kk] = 1;
            }
        }
    }

    filter_size = 1;
    for(jj = 0; jj < rank; jj++)
        filter_size *= order + 1;

    if (!NI_InitPointIterator(output, &io))
        goto exit;

    pi = (char *)PyArray_DATA(input);
    po = (char *)PyArray_DATA(output);

    /* store all coordinates and offsets with filter: */
    fcoordinates = (npy_intp*)malloc(rank * filter_size * sizeof(npy_intp));
    foffsets = (npy_intp*)malloc(filter_size * sizeof(npy_intp));
    if (NPY_UNLIKELY(!fcoordinates || !foffsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }

    for(jj = 0; jj < rank; jj++)
        ftmp[jj] = 0;
    kk = 0;
    for(hh = 0; hh < filter_size; hh++) {
        for(jj = 0; jj < rank; jj++)
            fcoordinates[jj + hh * rank] = ftmp[jj];
        foffsets[hh] = kk;
        for(jj = rank - 1; jj >= 0; jj--) {
            if (ftmp[jj] < order) {
                ftmp[jj]++;
                kk += istrides[jj];
                break;
            } else {
                ftmp[jj] = 0;
                kk -= istrides[jj] * order;
            }
        }
    }
    size = PyArray_SIZE(output);
    for(kk = 0; kk < size; kk++) {
        double t = 0.0;
        npy_intp edge = 0, oo = 0, zero = 0;

        for(hh = 0; hh < rank; hh++) {
            if (zeros && zeros[hh][io.coordinates[hh]]) {
                /* we use constant border condition */
                zero = 1;
                break;
            }
            oo += offsets[hh][io.coordinates[hh]];
            if (edge_offsets[hh][io.coordinates[hh]])
                edge = 1;
        }

        if (!reverse) {
            // forward computation
            if (!zero) {
                npy_intp *ff = fcoordinates;
                const int type_num = PyArray_TYPE(input);
                t = 0.0;
                for(hh = 0; hh < filter_size; hh++) {
                    npy_intp idx = 0;
                    double coeff = 0.0;

                    if (NPY_UNLIKELY(edge)) {
                        /* use precalculated edge offsets: */
                        for(jj = 0; jj < rank; jj++) {
                            if (edge_offsets[jj][io.coordinates[jj]])
                                idx += edge_offsets[jj][io.coordinates[jj]][ff[jj]];
                            else
                                idx += ff[jj] * istrides[jj];
                        }
                        idx += oo;
                    } else {
                        /* use normal offsets: */
                        idx += oo + foffsets[hh];
                    }
                    switch (type_num) {
                        CASE_INTERP_COEFF(NPY_BOOL, npy_bool,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_UBYTE, npy_ubyte,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_USHORT, npy_ushort,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_UINT, npy_uint,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_ULONG, npy_ulong,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_ULONGLONG, npy_ulonglong,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_BYTE, npy_byte,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_SHORT, npy_short,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_INT, npy_int,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_LONG, npy_long,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_LONGLONG, npy_longlong,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_FLOAT, npy_float,
                                          coeff, pi, idx);
                        CASE_INTERP_COEFF(NPY_DOUBLE, npy_double,
                                          coeff, pi, idx);
                    default:
                        NPY_END_THREADS;
                        PyErr_SetString(PyExc_RuntimeError,
                                        "data type not supported");
                        goto exit;
                    }
                    /* calculate interpolated value: */
                    for(jj = 0; jj < rank; jj++)
                        if (order > 0)
                            coeff *= splvals[jj][io.coordinates[jj]][ff[jj]];
                    t += coeff;
                    ff += rank;
                }
            } else {
                t = cval;
            }
            /* store output: */
            switch (PyArray_TYPE(output)) {
                CASE_INTERP_OUT(NPY_BOOL, npy_bool, po, t);
                CASE_INTERP_OUT_UINT(UBYTE, npy_ubyte, po, t);
                CASE_INTERP_OUT_UINT(USHORT, npy_ushort, po, t);
                CASE_INTERP_OUT_UINT(UINT, npy_uint, po, t);
                CASE_INTERP_OUT_UINT(ULONG, npy_ulong, po, t);
                CASE_INTERP_OUT_UINT(ULONGLONG, npy_ulonglong, po, t);
                CASE_INTERP_OUT_INT(BYTE, npy_byte, po, t);
                CASE_INTERP_OUT_INT(SHORT, npy_short, po, t);
                CASE_INTERP_OUT_INT(INT, npy_int, po, t);
                CASE_INTERP_OUT_INT(LONG, npy_long, po, t);
                CASE_INTERP_OUT_INT(LONGLONG, npy_longlong, po, t);
                CASE_INTERP_OUT(NPY_FLOAT, npy_float, po, t);
                CASE_INTERP_OUT(NPY_DOUBLE, npy_double, po, t);
            default:
                NPY_END_THREADS;
                PyErr_SetString(PyExc_RuntimeError, "data type not supported");
                goto exit;
            }
        } else {
            // this branch added for Theano:
            // computing the gradient in input based on the values from output
            if (!zero) {
                /* fetch output gradient: */
                double grad = 0.0;
                switch (PyArray_TYPE(output)) {
                    CASE_INTERP_GRAD(BOOL, npy_bool, grad, po);
                    CASE_INTERP_GRAD(UBYTE, npy_ubyte, grad, po);
                    CASE_INTERP_GRAD(USHORT, npy_ushort, grad, po);
                    CASE_INTERP_GRAD(UINT, npy_uint, grad, po);
                    CASE_INTERP_GRAD(ULONG, npy_ulong, grad, po);
                    CASE_INTERP_GRAD(ULONGLONG, npy_ulonglong, grad, po);
                    CASE_INTERP_GRAD(BYTE, npy_byte, grad, po);
                    CASE_INTERP_GRAD(SHORT, npy_short, grad, po);
                    CASE_INTERP_GRAD(INT, npy_int, grad, po);
                    CASE_INTERP_GRAD(LONG, npy_long, grad, po);
                    CASE_INTERP_GRAD(LONGLONG, npy_longlong, grad, po);
                    CASE_INTERP_GRAD(FLOAT, npy_float, grad, po);
                    CASE_INTERP_GRAD(DOUBLE, npy_double, grad, po);
                default:
                    NPY_END_THREADS;
                    PyErr_SetString(PyExc_RuntimeError, "data type not supported");
                    goto exit;
                }

                npy_intp *ff = fcoordinates;
                const int type_num = PyArray_TYPE(input);
                for(hh = 0; hh < filter_size; hh++) {
                    npy_intp idx = 0;
                    double coeff = grad;
                    /* calculate interpolated value: */
                    for(jj = 0; jj < rank; jj++)
                        if (order > 0)
                            coeff *= splvals[jj][io.coordinates[jj]][ff[jj]];

                    if (NPY_UNLIKELY(edge)) {
                        /* use precalculated edge offsets: */
                        for(jj = 0; jj < rank; jj++) {
                            if (edge_offsets[jj][io.coordinates[jj]])
                                idx += edge_offsets[jj][io.coordinates[jj]][ff[jj]];
                            else
                                idx += ff[jj] * istrides[jj];
                        }
                        idx += oo;
                    } else {
                        /* use normal offsets: */
                        idx += oo + foffsets[hh];
                    }
                    switch (type_num) {
                        CASE_INTERP_INCR(NPY_BOOL, npy_bool, (pi + idx), coeff);
                        CASE_INTERP_INCR_UINT(UBYTE, npy_ubyte, (pi + idx), coeff);
                        CASE_INTERP_INCR_UINT(USHORT, npy_ushort, (pi + idx), coeff);
                        CASE_INTERP_INCR_UINT(UINT, npy_uint, (pi + idx), coeff);
                        CASE_INTERP_INCR_UINT(ULONG, npy_ulong, (pi + idx), coeff);
                        CASE_INTERP_INCR_UINT(ULONGLONG, npy_ulonglong, (pi + idx), coeff);
                        CASE_INTERP_INCR_INT(BYTE, npy_byte, (pi + idx), coeff);
                        CASE_INTERP_INCR_INT(SHORT, npy_short, (pi + idx), coeff);
                        CASE_INTERP_INCR_INT(INT, npy_int, (pi + idx), coeff);
                        CASE_INTERP_INCR_INT(LONG, npy_long, (pi + idx), coeff);
                        CASE_INTERP_INCR_INT(LONGLONG, npy_longlong, (pi + idx), coeff);
                        CASE_INTERP_INCR(NPY_FLOAT, npy_float, (pi + idx), coeff);
                        CASE_INTERP_INCR(NPY_DOUBLE, npy_double, (pi + idx), coeff);
                    default:
                        NPY_END_THREADS;
                        PyErr_SetString(PyExc_RuntimeError,
                                        "data type not supported");
                        goto exit;
                    }
                    ff += rank;
                }
            }
        }
        NI_ITERATOR_NEXT(io, po);
    }

 exit:
    NPY_END_THREADS;
    if (zeros) {
        for(jj = 0; jj < rank; jj++)
            free(zeros[jj]);
        free(zeros);
    }
    if (offsets) {
        for(jj = 0; jj < rank; jj++)
            free(offsets[jj]);
        free(offsets);
    }
    if (splvals) {
        for(jj = 0; jj < rank; jj++) {
            if (splvals[jj]) {
                for(hh = 0; hh < odimensions[jj]; hh++)
                    free(splvals[jj][hh]);
                free(splvals[jj]);
            }
        }
        free(splvals);
    }
    if (edge_offsets) {
        for(jj = 0; jj < rank; jj++) {
            if (edge_offsets[jj]) {
                for(hh = 0; hh < odimensions[jj]; hh++)
                    free(edge_offsets[jj][hh]);
                free(edge_offsets[jj]);
            }
        }
        free(edge_offsets);
    }
    free(foffsets);
    free(fcoordinates);
    return PyErr_Occurred() ? 0 : 1;
}
