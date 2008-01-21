
#ifndef _OMEGA_H
#define _OMEGA_H

//#include whatever defines PyArrayObject

template<typename T>
struct TMat_t
{
  T *  __restrict__ d;/**< pointer to element (0,0) */
  size_t    M;    /**< number of rows */
  size_t    N;    /**< number of columns */
  size_t    m;    /**< row stride */
  size_t    n;    /**< column stride */
  bool invalid;

  /** null  */
TMat_t(const PyArrayObject *o) : 
  d((double*) o->data),
    M((o->nd==2) ? o->dimensions[0] : 0),
    N((o->nd==2) ? o->dimensions[1] : 0),
    m((o->nd==2) ? o->strides[0] / sizeof(double) : 0),
    n((o->nd==2) ? o->strides[1] / sizeof(double) : 0),
    invalid((o->nd !=2) || (o->descr->elsize != sizeof(T)))
  {
  }
  /** unsafe element access */
  const T & operator()(size_t i, size_t j) const
  {
    return d[ i * m + j*n];
  }
  /** unsafe element access */
  T & operator()(size_t i, size_t j)
  {
    return d[ i * m + j*n];
  }
  /** safe element access */
  const T & at(size_t i, size_t j) const
  {
    return d[ assert((i < M) && (j < N)), i * m + j*n];
  }
  /** safe element access */
  T & at(size_t i, size_t j)
  {
    return d[ assert((i < M) && (j < N)), i * m + j*n];
  }
};

#endif
