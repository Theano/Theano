""" Header text for the C and Fortran BLAS interfaces.

There is no standard name or location for this header, so we just insert it ourselves into the C code
"""
def cblas_header_text():
    """C header for the cblas interface."""

    return """
    //#include <stddef.h>

    #undef __BEGIN_DECLS
    #undef __END_DECLS
    #ifdef __cplusplus
    #define __BEGIN_DECLS extern "C" {
    #define __END_DECLS }
    #else
    #define __BEGIN_DECLS           /* empty */
    #define __END_DECLS             /* empty */
    #endif

    __BEGIN_DECLS

    #define MOD %

    /*
     * Enumerated and derived types
     */
    #define CBLAS_INDEX size_t  /* this may vary between platforms */

    enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
    enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
    enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
    enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
    enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

    float  cblas_sdsdot(const int N, const float alpha, const float *X,
                        const int incX, const float *Y, const int incY);
    double cblas_dsdot(const int N, const float *X, const int incX, const float *Y,
                       const int incY);
    float  cblas_sdot(const int N, const float  *X, const int incX,
                      const float  *Y, const int incY);
    double cblas_ddot(const int N, const double *X, const int incX,
                      const double *Y, const int incY);

    /*
     * Functions having prefixes Z and C only
     */
    void   cblas_cdotu_sub(const int N, const void *X, const int incX,
                           const void *Y, const int incY, void *dotu);
    void   cblas_cdotc_sub(const int N, const void *X, const int incX,
                           const void *Y, const int incY, void *dotc);

    void   cblas_zdotu_sub(const int N, const void *X, const int incX,
                           const void *Y, const int incY, void *dotu);
    void   cblas_zdotc_sub(const int N, const void *X, const int incX,
                           const void *Y, const int incY, void *dotc);


    /*
     * Functions having prefixes S D SC DZ
     */
    float  cblas_snrm2(const int N, const float *X, const int incX);
    float  cblas_sasum(const int N, const float *X, const int incX);

    double cblas_dnrm2(const int N, const double *X, const int incX);
    double cblas_dasum(const int N, const double *X, const int incX);

    float  cblas_scnrm2(const int N, const void *X, const int incX);
    float  cblas_scasum(const int N, const void *X, const int incX);

    double cblas_dznrm2(const int N, const void *X, const int incX);
    double cblas_dzasum(const int N, const void *X, const int incX);


    /*
     * Functions having standard 4 prefixes (S D C Z)
     */
    CBLAS_INDEX cblas_isamax(const int N, const float  *X, const int incX);
    CBLAS_INDEX cblas_idamax(const int N, const double *X, const int incX);
    CBLAS_INDEX cblas_icamax(const int N, const void   *X, const int incX);
    CBLAS_INDEX cblas_izamax(const int N, const void   *X, const int incX);

    /*
     * ===========================================================================
     * Prototypes for level 1 BLAS routines
     * ===========================================================================
     */

    /* 
     * Routines with standard 4 prefixes (s, d, c, z)
     */
    void cblas_sswap(const int N, float *X, const int incX, 
                     float *Y, const int incY);
    void cblas_scopy(const int N, const float *X, const int incX, 
                     float *Y, const int incY);
    void cblas_saxpy(const int N, const float alpha, const float *X,
                     const int incX, float *Y, const int incY);

    void cblas_dswap(const int N, double *X, const int incX, 
                     double *Y, const int incY);
    void cblas_dcopy(const int N, const double *X, const int incX, 
                     double *Y, const int incY);
    void cblas_daxpy(const int N, const double alpha, const double *X,
                     const int incX, double *Y, const int incY);

    void cblas_cswap(const int N, void *X, const int incX, 
                     void *Y, const int incY);
    void cblas_ccopy(const int N, const void *X, const int incX, 
                     void *Y, const int incY);
    void cblas_caxpy(const int N, const void *alpha, const void *X,
                     const int incX, void *Y, const int incY);

    void cblas_zswap(const int N, void *X, const int incX, 
                     void *Y, const int incY);
    void cblas_zcopy(const int N, const void *X, const int incX, 
                     void *Y, const int incY);
    void cblas_zaxpy(const int N, const void *alpha, const void *X,
                     const int incX, void *Y, const int incY);


    /* 
     * Routines with S and D prefix only
     */
    void cblas_srotg(float *a, float *b, float *c, float *s);
    void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
    void cblas_srot(const int N, float *X, const int incX,
                    float *Y, const int incY, const float c, const float s);
    void cblas_srotm(const int N, float *X, const int incX,
                    float *Y, const int incY, const float *P);

    void cblas_drotg(double *a, double *b, double *c, double *s);
    void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
    void cblas_drot(const int N, double *X, const int incX,
                    double *Y, const int incY, const double c, const double  s);
    void cblas_drotm(const int N, double *X, const int incX,
                    double *Y, const int incY, const double *P);


    /* 
     * Routines with S D C Z CS and ZD prefixes
     */
    void cblas_sscal(const int N, const float alpha, float *X, const int incX);
    void cblas_dscal(const int N, const double alpha, double *X, const int incX);
    void cblas_cscal(const int N, const void *alpha, void *X, const int incX);
    void cblas_zscal(const int N, const void *alpha, void *X, const int incX);
    void cblas_csscal(const int N, const float alpha, void *X, const int incX);
    void cblas_zdscal(const int N, const double alpha, void *X, const int incX);

    /*
     * ===========================================================================
     * Prototypes for level 2 BLAS
     * ===========================================================================
     */

    /* 
     * Routines with standard 4 prefixes (S, D, C, Z)
     */
    void cblas_sgemv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const float alpha, const float *A, const int lda,
                     const float *X, const int incX, const float beta,
                     float *Y, const int incY);
    void cblas_sgbmv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const int KL, const int KU, const float alpha,
                     const float *A, const int lda, const float *X,
                     const int incX, const float beta, float *Y, const int incY);
    void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const float *A, const int lda, 
                     float *X, const int incX);
    void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const int K, const float *A, const int lda, 
                     float *X, const int incX);
    void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const float *Ap, float *X, const int incX);
    void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const float *A, const int lda, float *X,
                     const int incX);
    void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const int K, const float *A, const int lda,
                     float *X, const int incX);
    void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const float *Ap, float *X, const int incX);

    void cblas_dgemv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     const double *X, const int incX, const double beta,
                     double *Y, const int incY);
    void cblas_dgbmv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const int KL, const int KU, const double alpha,
                     const double *A, const int lda, const double *X,
                     const int incX, const double beta, double *Y, const int incY);
    void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const double *A, const int lda, 
                     double *X, const int incX);
    void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const int K, const double *A, const int lda, 
                     double *X, const int incX);
    void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const double *Ap, double *X, const int incX);
    void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const double *A, const int lda, double *X,
                     const int incX);
    void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const int K, const double *A, const int lda,
                     double *X, const int incX);
    void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const double *Ap, double *X, const int incX);

    void cblas_cgemv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     const void *X, const int incX, const void *beta,
                     void *Y, const int incY);
    void cblas_cgbmv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const int KL, const int KU, const void *alpha,
                     const void *A, const int lda, const void *X,
                     const int incX, const void *beta, void *Y, const int incY);
    void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const void *A, const int lda, 
                     void *X, const int incX);
    void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const int K, const void *A, const int lda, 
                     void *X, const int incX);
    void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const void *Ap, void *X, const int incX);
    void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const void *A, const int lda, void *X,
                     const int incX);
    void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const int K, const void *A, const int lda,
                     void *X, const int incX);
    void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const void *Ap, void *X, const int incX);

    void cblas_zgemv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     const void *X, const int incX, const void *beta,
                     void *Y, const int incY);
    void cblas_zgbmv(const enum CBLAS_ORDER order,
                     const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                     const int KL, const int KU, const void *alpha,
                     const void *A, const int lda, const void *X,
                     const int incX, const void *beta, void *Y, const int incY);
    void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const void *A, const int lda, 
                     void *X, const int incX);
    void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const int K, const void *A, const int lda, 
                     void *X, const int incX);
    void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const void *Ap, void *X, const int incX);
    void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const void *A, const int lda, void *X,
                     const int incX);
    void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const int K, const void *A, const int lda,
                     void *X, const int incX);
    void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                     const int N, const void *Ap, void *X, const int incX);


    /* 
     * Routines with S and D prefixes only
     */
    void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const float alpha, const float *A,
                     const int lda, const float *X, const int incX,
                     const float beta, float *Y, const int incY);
    void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const int K, const float alpha, const float *A,
                     const int lda, const float *X, const int incX,
                     const float beta, float *Y, const int incY);
    void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const float alpha, const float *Ap,
                     const float *X, const int incX,
                     const float beta, float *Y, const int incY);
    void cblas_sger(const enum CBLAS_ORDER order, const int M, const int N,
                    const float alpha, const float *X, const int incX,
                    const float *Y, const int incY, float *A, const int lda);
    void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const float alpha, const float *X,
                    const int incX, float *A, const int lda);
    void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const float alpha, const float *X,
                    const int incX, float *Ap);
    void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const float alpha, const float *X,
                    const int incX, const float *Y, const int incY, float *A,
                    const int lda);
    void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const float alpha, const float *X,
                    const int incX, const float *Y, const int incY, float *A);

    void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const double alpha, const double *A,
                     const int lda, const double *X, const int incX,
                     const double beta, double *Y, const int incY);
    void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const int K, const double alpha, const double *A,
                     const int lda, const double *X, const int incX,
                     const double beta, double *Y, const int incY);
    void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const double alpha, const double *Ap,
                     const double *X, const int incX,
                     const double beta, double *Y, const int incY);
    void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
                    const double alpha, const double *X, const int incX,
                    const double *Y, const int incY, double *A, const int lda);
    void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const double alpha, const double *X,
                    const int incX, double *A, const int lda);
    void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const double alpha, const double *X,
                    const int incX, double *Ap);
    void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const double alpha, const double *X,
                    const int incX, const double *Y, const int incY, double *A,
                    const int lda);
    void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const double alpha, const double *X,
                    const int incX, const double *Y, const int incY, double *A);


    /* 
     * Routines with C and Z prefixes only
     */
    void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const void *alpha, const void *A,
                     const int lda, const void *X, const int incX,
                     const void *beta, void *Y, const int incY);
    void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const int K, const void *alpha, const void *A,
                     const int lda, const void *X, const int incX,
                     const void *beta, void *Y, const int incY);
    void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const void *alpha, const void *Ap,
                     const void *X, const int incX,
                     const void *beta, void *Y, const int incY);
    void cblas_cgeru(const enum CBLAS_ORDER order, const int M, const int N,
                     const void *alpha, const void *X, const int incX,
                     const void *Y, const int incY, void *A, const int lda);
    void cblas_cgerc(const enum CBLAS_ORDER order, const int M, const int N,
                     const void *alpha, const void *X, const int incX,
                     const void *Y, const int incY, void *A, const int lda);
    void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const float alpha, const void *X, const int incX,
                    void *A, const int lda);
    void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const float alpha, const void *X,
                    const int incX, void *A);
    void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                    const void *alpha, const void *X, const int incX,
                    const void *Y, const int incY, void *A, const int lda);
    void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                    const void *alpha, const void *X, const int incX,
                    const void *Y, const int incY, void *Ap);

    void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const void *alpha, const void *A,
                     const int lda, const void *X, const int incX,
                     const void *beta, void *Y, const int incY);
    void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const int K, const void *alpha, const void *A,
                     const int lda, const void *X, const int incX,
                     const void *beta, void *Y, const int incY);
    void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                     const int N, const void *alpha, const void *Ap,
                     const void *X, const int incX,
                     const void *beta, void *Y, const int incY);
    void cblas_zgeru(const enum CBLAS_ORDER order, const int M, const int N,
                     const void *alpha, const void *X, const int incX,
                     const void *Y, const int incY, void *A, const int lda);
    void cblas_zgerc(const enum CBLAS_ORDER order, const int M, const int N,
                     const void *alpha, const void *X, const int incX,
                     const void *Y, const int incY, void *A, const int lda);
    void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const double alpha, const void *X, const int incX,
                    void *A, const int lda);
    void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                    const int N, const double alpha, const void *X,
                    const int incX, void *A);
    void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                    const void *alpha, const void *X, const int incX,
                    const void *Y, const int incY, void *A, const int lda);
    void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                    const void *alpha, const void *X, const int incX,
                    const void *Y, const int incY, void *Ap);

    /*
     * ===========================================================================
     * Prototypes for level 3 BLAS
     * ===========================================================================
     */

    /* 
     * Routines with standard 4 prefixes (S, D, C, Z)
     */
    void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const int lda, const float *B, const int ldb,
                     const float beta, float *C, const int ldc);
    void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const int M, const int N,
                     const float alpha, const float *A, const int lda,
                     const float *B, const int ldb, const float beta,
                     float *C, const int ldc);
    void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                     const float alpha, const float *A, const int lda,
                     const float beta, float *C, const int ldc);
    void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                      const float alpha, const float *A, const int lda,
                      const float *B, const int ldb, const float beta,
                      float *C, const int ldc);
    void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_DIAG Diag, const int M, const int N,
                     const float alpha, const float *A, const int lda,
                     float *B, const int ldb);
    void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_DIAG Diag, const int M, const int N,
                     const float alpha, const float *A, const int lda,
                     float *B, const int ldb);

    void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const double alpha, const double *A,
                     const int lda, const double *B, const int ldb,
                     const double beta, double *C, const int ldc);
    void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     const double *B, const int ldb, const double beta,
                     double *C, const int ldc);
    void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                     const double alpha, const double *A, const int lda,
                     const double beta, double *C, const int ldc);
    void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                      const double alpha, const double *A, const int lda,
                      const double *B, const int ldb, const double beta,
                      double *C, const int ldc);
    void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_DIAG Diag, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     double *B, const int ldb);
    void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_DIAG Diag, const int M, const int N,
                     const double alpha, const double *A, const int lda,
                     double *B, const int ldb);

    void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const void *alpha, const void *A,
                     const int lda, const void *B, const int ldb,
                     const void *beta, void *C, const int ldc);
    void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     const void *B, const int ldb, const void *beta,
                     void *C, const int ldc);
    void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                     const void *alpha, const void *A, const int lda,
                     const void *beta, void *C, const int ldc);
    void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                      const void *alpha, const void *A, const int lda,
                      const void *B, const int ldb, const void *beta,
                      void *C, const int ldc);
    void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_DIAG Diag, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     void *B, const int ldb);
    void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_DIAG Diag, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     void *B, const int ldb);

    void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const void *alpha, const void *A,
                     const int lda, const void *B, const int ldb,
                     const void *beta, void *C, const int ldc);
    void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     const void *B, const int ldb, const void *beta,
                     void *C, const int ldc);
    void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                     const void *alpha, const void *A, const int lda,
                     const void *beta, void *C, const int ldc);
    void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                      const void *alpha, const void *A, const int lda,
                      const void *B, const int ldb, const void *beta,
                      void *C, const int ldc);
    void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_DIAG Diag, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     void *B, const int ldb);
    void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_DIAG Diag, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     void *B, const int ldb);


    /* 
     * Routines with prefixes C and Z only
     */
    void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     const void *B, const int ldb, const void *beta,
                     void *C, const int ldc);
    void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                     const float alpha, const void *A, const int lda,
                     const float beta, void *C, const int ldc);
    void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                      const void *alpha, const void *A, const int lda,
                      const void *B, const int ldb, const float beta,
                      void *C, const int ldc);

    void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                     const enum CBLAS_UPLO Uplo, const int M, const int N,
                     const void *alpha, const void *A, const int lda,
                     const void *B, const int ldb, const void *beta,
                     void *C, const int ldc);
    void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                     const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                     const double alpha, const void *A, const int lda,
                     const double beta, void *C, const int ldc);
    void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                      const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                      const void *alpha, const void *A, const int lda,
                      const void *B, const int ldb, const double beta,
                      void *C, const int ldc);

    void cblas_xerbla(int p, const char *rout, const char *form, ...);

    __END_DECLS
    """

def blas_header_text():
    """C header for the fortran blas interface"""
    return """
    extern "C"
    {

        void xerbla_(char*, void *);

    /***********/
    /* Level 1 */
    /***********/

    /* Single Precision */

        void srot_(const int*, float *, const int*, float *, const int*, const float *, const float *);
        void srotg_(float *,float *,float *,float *);    
        void srotm_( const int*, float *, const int*, float *, const int*, const float *);
        void srotmg_(float *,float *,float *,const float *, float *);
        void sswap_( const int*, float *, const int*, float *, const int*);
        void scopy_( const int*, const float *, const int*, float *, const int*);
        void saxpy_( const int*, const float *, const float *, const int*, float *, const int*);
        void sdot_sub_(const int*, const float *, const int*, const float *, const int*, float *);
        void sdsdot_sub_( const int*, const float *, const float *, const int*, const float *, const int*, float *);
        void sscal_( const int*, const float *, float *, const int*);
        void snrm2_sub_( const int*, const float *, const int*, float *);
        void sasum_sub_( const int*, const float *, const int*, float *);
        void isamax_sub_( const int*, const float * , const int*, const int*);

    /* Double Precision */

        void drot_(const int*, double *, const int*, double *, const int*, const double *, const double *);
        void drotg_(double *,double *,double *,double *);    
        void drotm_( const int*, double *, const int*, double *, const int*, const double *);
        void drotmg_(double *,double *,double *,const double *, double *);
        void dswap_( const int*, double *, const int*, double *, const int*);
        void dcopy_( const int*, const double *, const int*, double *, const int*);
        void daxpy_( const int*, const double *, const double *, const int*, double *, const int*);
        void dswap_( const int*, double *, const int*, double *, const int*);
        void dsdot_sub_(const int*, const float *, const int*, const float *, const int*, double *);
        void ddot_sub_( const int*, const double *, const int*, const double *, const int*, double *);
        void dscal_( const int*, const double *, double *, const int*);
        void dnrm2_sub_( const int*, const double *, const int*, double *);
        void dasum_sub_( const int*, const double *, const int*, double *);
        void idamax_sub_( const int*, const double * , const int*, const int*);

    /* Single Complex Precision */

        void cswap_( const int*, void *, const int*, void *, const int*);
        void ccopy_( const int*, const void *, const int*, void *, const int*);
        void caxpy_( const int*, const void *, const void *, const int*, void *, const int*);
        void cswap_( const int*, void *, const int*, void *, const int*);
        void cdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void cdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void cscal_( const int*, const void *, void *, const int*);
        void icamax_sub_( const int*, const void *, const int*, const int*);
        void csscal_( const int*, const float *, void *, const int*);
        void scnrm2_sub_( const int*, const void *, const int*, float *);
        void scasum_sub_( const int*, const void *, const int*, float *);

    /* Double Complex Precision */

        void zswap_( const int*, void *, const int*, void *, const int*);
        void zcopy_( const int*, const void *, const int*, void *, const int*);
        void zaxpy_( const int*, const void *, const void *, const int*, void *, const int*);
        void zswap_( const int*, void *, const int*, void *, const int*);
        void zdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void zdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
        void zdscal_( const int*, const double *, void *, const int*);
        void zscal_( const int*, const void *, void *, const int*);
        void dznrm2_sub_( const int*, const void *, const int*, double *);
        void dzasum_sub_( const int*, const void *, const int*, double *);
        void izamax_sub_( const int*, const void *, const int*, const int*);

    /***********/
    /* Level 2 */
    /***********/

    /* Single Precision */

        void sgemv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void sgbmv_(char*, const int*, const int*, const int*, const int*, const float *,  const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssymv_(char*, const int*, const float *, const float *, const int*, const float *,  const int*, const float *, float *, const int*);
        void ssbmv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void sspmv_(char*, const int*, const float *, const float *, const float *, const int*, const float *, float *, const int*);
        void strmv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
        void stbmv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
        void strsv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
        void stbsv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
        void stpmv_( char*, char*, char*, const int*, const float *, float *, const int*);
        void stpsv_( char*, char*, char*, const int*, const float *, float *, const int*);
        void sger_( const int*, const int*, const float *, const float *, const int*, const float *, const int*, float *, const int*);
        void ssyr_(char*, const int*, const float *, const float *, const int*, float *, const int*);
        void sspr_(char*, const int*, const float *, const float *, const int*, float *); 
        void sspr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *); 
        void ssyr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *, const int*);

    /* Double Precision */

        void dgemv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dgbmv_(char*, const int*, const int*, const int*, const int*, const double *,  const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsymv_(char*, const int*, const double *, const double *, const int*, const double *,  const int*, const double *, double *, const int*);
        void dsbmv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dspmv_(char*, const int*, const double *, const double *, const double *, const int*, const double *, double *, const int*);
        void dtrmv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
        void dtbmv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
        void dtrsv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
        void dtbsv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
        void dtpmv_( char*, char*, char*, const int*, const double *, double *, const int*);
        void dtpsv_( char*, char*, char*, const int*, const double *, double *, const int*);
        void dger_( const int*, const int*, const double *, const double *, const int*, const double *, const int*, double *, const int*);
        void dsyr_(char*, const int*, const double *, const double *, const int*, double *, const int*);
        void dspr_(char*, const int*, const double *, const double *, const int*, double *); 
        void dspr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *); 
        void dsyr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *, const int*);

    /* Single Complex Precision */

        void cgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void cgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void chpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
        void ctrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ctbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ctpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
        void ctrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ctbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ctpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
        void cgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void cgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
        void cher_(char*, const int*, const float *, const void *, const int*, void *, const int*);
        void cher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void chpr_(char*, const int*, const float *, const void *, const int*, void *);
        void chpr2_(char*, const int*, const float *, const void *, const int*, const void *, const int*, void *);

    /* Double Complex Precision */

        void zgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
        void zhpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
        void ztrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ztbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ztpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
        void ztrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
        void ztbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
        void ztpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
        void zgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void zgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
        void zher_(char*, const int*, const double *, const void *, const int*, void *, const int*);
        void zher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
        void zhpr_(char*, const int*, const double *, const void *, const int*, void *);
        void zhpr2_(char*, const int*, const double *, const void *, const int*, const void *, const int*, void *);

    /***********/
    /* Level 3 */
    /***********/

    /* Single Precision */

        void sgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ssyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void ssyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void strmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
        void strsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

    /* Double Precision */

        void dgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void dsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void dtrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
        void dtrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

    /* Single Complex Precision */

        void cgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void csymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void chemm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void csyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void cherk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
        void csyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void cher2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
        void ctrmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
        void ctrsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

    /* Double Complex Precision */

        void zgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zhemm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void zherk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
        void zsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void zher2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
        void ztrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
        void ztrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

    }
    """

def ____gemm_code(check_ab, a_init, b_init):
    mod = '%'
    return """
        const char * error_string = NULL;

        int type_num = _x->descr->type_num;
        int type_size = _x->descr->elsize; // in bytes

        npy_intp* Nx = _x->dimensions;
        npy_intp* Ny = _y->dimensions;
        npy_intp* Nz = _z->dimensions;

        npy_intp* Sx = _x->strides;
        npy_intp* Sy = _y->strides;
        npy_intp* Sz = _z->strides;

        size_t sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;

        int unit = 0;

        if (_x->nd != 2) goto _dot_execute_fallback;
        if (_y->nd != 2) goto _dot_execute_fallback;
        if (_z->nd != 2) goto _dot_execute_fallback;

        %(check_ab)s

        if ((_x->descr->type_num != PyArray_DOUBLE) 
            && (_x->descr->type_num != PyArray_FLOAT))
            goto _dot_execute_fallback;

        if ((_y->descr->type_num != PyArray_DOUBLE) 
            && (_y->descr->type_num != PyArray_FLOAT))
            goto _dot_execute_fallback;

        if ((_y->descr->type_num != PyArray_DOUBLE) 
            && (_y->descr->type_num != PyArray_FLOAT))
            goto _dot_execute_fallback;

        if ((_x->descr->type_num != _y->descr->type_num)
            ||(_x->descr->type_num != _z->descr->type_num))
            goto _dot_execute_fallback;


        if ((Nx[0] != Nz[0]) || (Nx[1] != Ny[0]) || (Ny[1] != Nz[1]))
        {
            error_string = "Input dimensions do not agree";
            goto _dot_execute_fail;
        }
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] %(mod)s type_size) || (Sx[1] %(mod)s type_size)
           || (Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] %(mod)s type_size) || (Sy[1] %(mod)s type_size)
           || (Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] %(mod)s type_size) || (Sz[1] %(mod)s type_size))
        {
           goto _dot_execute_fallback;
        }

        /*
        encode the stride structure of _x,_y,_z into a single integer
        */
        unit |= ((Sx[1] == type_size) ? 0x0 : (Sx[0] == type_size) ? 0x1 : 0x2) << 0;
        unit |= ((Sy[1] == type_size) ? 0x0 : (Sy[0] == type_size) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size) ? 0x0 : (Sz[0] == type_size) ? 0x1 : 0x2) << 8;

        /* create appropriate strides for malformed matrices that are row or column
         * vectors
         */
        sx_0 = (Nx[0] > 1) ? Sx[0]/type_size : Nx[1];
        sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : Nx[0];
        sy_0 = (Ny[0] > 1) ? Sy[0]/type_size : Ny[1];
        sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : Ny[0];
        sz_0 = (Nz[0] > 1) ? Sz[0]/type_size : Nz[1];
        sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : Nz[0];

        switch (type_num)
        {
            case PyArray_FLOAT:
            {
                #define REAL float
                float a = %(a_init)s;
                float b = %(b_init)s;

                float* x = (float*)PyArray_DATA(_x);
                float* y = (float*)PyArray_DATA(_y);
                float* z = (float*)PyArray_DATA(_z);

                switch(unit)
                {
                    case 0x000: cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_0, b, z, sz_0); break;
                    case 0x001: cblas_sgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_0, b, z, sz_0); break;
                    case 0x010: cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_1, b, z, sz_0); break;
                    case 0x011: cblas_sgemm(CblasRowMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_1, b, z, sz_0); break;
                    case 0x100: cblas_sgemm(CblasColMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_0, b, z, sz_1); break;
                    case 0x101: cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_0, b, z, sz_1); break;
                    case 0x110: cblas_sgemm(CblasColMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_1, b, z, sz_1); break;
                    case 0x111: cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_1, b, z, sz_1); break;
                    default: goto _dot_execute_fallback;
                };
                #undef REAL
            }
            break;
            case PyArray_DOUBLE:
            {
                #define REAL double
                double a = %(a_init)s;
                double b = %(b_init)s;

                double* x = (double*)PyArray_DATA(_x);
                double* y = (double*)PyArray_DATA(_y);
                double* z = (double*)PyArray_DATA(_z);
                switch(unit)
                {
                    case 0x000: cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_0, b, z, sz_0); break;
                    case 0x001: cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_0, b, z, sz_0); break;
                    case 0x010: cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_1, b, z, sz_0); break;
                    case 0x011: cblas_dgemm(CblasRowMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_1, b, z, sz_0); break;
                    case 0x100: cblas_dgemm(CblasColMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_0, b, z, sz_1); break;
                    case 0x101: cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_0, b, z, sz_1); break;
                    case 0x110: cblas_dgemm(CblasColMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_1, b, z, sz_1); break;
                    case 0x111: cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_1, b, z, sz_1); break;
                    default: goto _dot_execute_fallback;
                };
                #undef REAL
            }
            break;
        }

        return 0;  //success!

        _dot_execute_fallback:
        PyErr_SetString(PyExc_NotImplementedError, 
            "dot->execute() fallback");
        return -1;

        _dot_execute_fail:
        if (error_string == NULL)
            PyErr_SetString(PyExc_ValueError, 
                "dot->execute() cant run on these inputs");
        return -1;

        /* v 1 */
    """ % locals()

# currently unused, preferring the fallback method (throwing
# NotImplementedError) for when gemm won't work.
_templated_memaligned_gemm = """
template <typename Ta, typename Tx, typename Ty, typename Tb, typename Tz>
int general_gemm(int zM, int zN, int xN,.
    Ta a,
    Tx * x, int xm, int xn,
    Tx * y, int ym, int yn,
    Tb b,
    Tz * z, int zm, int zn)
{
    for (int i = 0; i < zM; ++i)
    {
        for (int j = 0; j < zN; ++j)
        {
            Tz zij = 0.0;
            for (int k = 0; k < xN; ++k)
            {
                zij += x[i*xm+k*xn] * y[k*ym+j*yn];
            }
            z[i * zm + j * zn] *= b;
            z[i * zm + j * zn] += a * zij;
        }
    }
}
"""

