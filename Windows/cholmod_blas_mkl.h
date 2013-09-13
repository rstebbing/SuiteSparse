#ifndef CHOLMOD_BLAS_MKL_H
#define CHOLMOD_BLAS_MKL_H

#include <mkl.h>
#include <stdlib.h>

#define CHOLMOD_ARCHITECTURE "Microsoft Windows"
#define BLAS_NO_UNDERSCORE
#define BLAS_INT int

#define CHECK_BLAS_INT (sizeof (BLAS_INT) < sizeof (Int))
#define EQ(K,k) (((BLAS_INT) K) == ((Int) k))

#define MKCOMPLEX(ptr) (MKL_Complex16 *)(ptr)

#define BLAS_DTRSV dtrsv
#define BLAS_DGEMV dgemv
#define BLAS_DTRSM dtrsm
#define BLAS_DGEMM dgemm
#define BLAS_DSYRK dsyrk
#define BLAS_DGER  dger
#define BLAS_DSCAL dscal
#define LAPACK_DPOTRF dpotrf

#define BLAS_ZTRSV ztrsv
#define BLAS_ZGEMV zgemv
#define BLAS_ZTRSM ztrsm
#define BLAS_ZGEMM zgemm
#define BLAS_ZHERK zherk
#define BLAS_ZGER  zgeru
#define BLAS_ZSCAL zscal
#define LAPACK_ZPOTRF zpotrf

#define CHECK_BLAS_INT (sizeof (BLAS_INT) < sizeof (Int))
#define EQ(K,k) (((BLAS_INT) K) == ((Int) k))

/* ========================================================================== */
/* === BLAS and LAPACK prototypes and macros ================================ */
/* ========================================================================== */
#define BLAS_dgemv(trans,m,n,alpha,A,lda,X,incx,beta,Y,incy) \
{ \
    BLAS_INT M = m, N = n, LDA = lda, INCX = incx, INCY = incy ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (N,n) && EQ (LDA,lda) && \
        EQ (INCX,incx) && EQ (INCY,incy))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_DGEMV (trans, &M, &N, alpha, A, &LDA, X, &INCX, beta, Y, &INCY) ; \
    } \
}

#define BLAS_zgemv(trans,m,n,alpha,A,lda,X,incx,beta,Y,incy) \
{ \
    BLAS_INT M = m, N = n, LDA = lda, INCX = incx, INCY = incy ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (N,n) && EQ (LDA,lda) && \
        EQ (INCX,incx) && EQ (INCY,incy))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_ZGEMV (trans, &M, &N, MKCOMPLEX(alpha), MKCOMPLEX(A), &LDA, MKCOMPLEX(X), &INCX, MKCOMPLEX(beta), MKCOMPLEX(Y), &INCY) ; \
    } \
}

#define BLAS_dtrsv(uplo,trans,diag,n,A,lda,X,incx) \
{ \
    BLAS_INT N = n, LDA = lda, INCX = incx ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ (LDA,lda) && EQ (INCX,incx))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_DTRSV (uplo, trans, diag, &N, A, &LDA, X, &INCX) ; \
    } \
}

#define BLAS_ztrsv(uplo,trans,diag,n,A,lda,X,incx) \
{ \
    BLAS_INT N = n, LDA = lda, INCX = incx ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ (LDA,lda) && EQ (INCX,incx))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_ZTRSV (uplo, trans, diag, &N, MKCOMPLEX(A), &LDA, MKCOMPLEX(X), &INCX) ; \
    } \
}

#define BLAS_dtrsm(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb) \
{ \
    BLAS_INT M = m, N = n, LDA = lda, LDB = ldb ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (N,n) && EQ (LDA,lda) && \
        EQ (LDB,ldb))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_DTRSM (side, uplo, transa, diag, &M, &N, alpha, A, &LDA, B, &LDB);\
    } \
}

#define BLAS_ztrsm(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb) \
{ \
    BLAS_INT M = m, N = n, LDA = lda, LDB = ldb ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (N,n) && EQ (LDA,lda) && \
        EQ (LDB,ldb))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_ZTRSM (side, uplo, transa, diag, &M, &N, MKCOMPLEX(alpha), MKCOMPLEX(A), &LDA, MKCOMPLEX(B), &LDB);\
    } \
}

#define BLAS_dgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc) \
{ \
    BLAS_INT M = m, N = n, K = k, LDA = lda, LDB = ldb, LDC = ldc ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (N,n) && EQ (K,k) && \
        EQ (LDA,lda) && EQ (LDB,ldb) && EQ (LDC,ldc))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_DGEMM (transa, transb, &M, &N, &K, alpha, A, &LDA, B, &LDB, beta, \
	    C, &LDC) ; \
    } \
}

#define BLAS_zgemm(transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc) \
{ \
    BLAS_INT M = m, N = n, K = k, LDA = lda, LDB = ldb, LDC = ldc ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (N,n) && EQ (K,k) && \
        EQ (LDA,lda) && EQ (LDB,ldb) && EQ (LDC,ldc))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_ZGEMM (transa, transb, &M, &N, &K, MKCOMPLEX(alpha), MKCOMPLEX(A), &LDA, MKCOMPLEX(B), &LDB, MKCOMPLEX(beta), \
	    MKCOMPLEX(C), &LDC) ; \
    } \
}

#define BLAS_dsyrk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc) \
{ \
    BLAS_INT N = n, K = k, LDA = lda, LDC = ldc ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ (K,k) && EQ (LDA,lda) && \
        EQ (LDC,ldc))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_DSYRK (uplo, trans, &N, &K, alpha, A, &LDA, beta, C, &LDC) ; \
    } \
} \

#define BLAS_zherk(uplo,trans,n,k,alpha,A,lda,beta,C,ldc) \
{ \
    BLAS_INT N = n, K = k, LDA = lda, LDC = ldc ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ (K,k) && EQ (LDA,lda) && \
        EQ (LDC,ldc))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_ZHERK (uplo, trans, &N, &K, alpha, MKCOMPLEX(A), &LDA, beta, MKCOMPLEX(C), &LDC) ; \
    } \
} \

#define LAPACK_dpotrf(uplo,n,A,lda,info) \
{ \
    BLAS_INT N = n, LDA = lda, INFO = 1 ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ (LDA,lda))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	LAPACK_DPOTRF (uplo, &N, A, &LDA, &INFO) ; \
    } \
    info = INFO ; \
}

#define LAPACK_zpotrf(uplo,n,A,lda,info) \
{ \
    BLAS_INT N = n, LDA = lda, INFO = 1 ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ (LDA,lda))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	LAPACK_ZPOTRF (uplo, &N, MKCOMPLEX(A), &LDA, &INFO) ; \
    } \
    info = INFO ; \
}

/* ========================================================================== */

#define BLAS_dscal(n,alpha,Y,incy) \
{ \
    BLAS_INT N = n, INCY = incy ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ (INCY,incy))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_DSCAL (&N, alpha, Y, &INCY) ; \
    } \
}

#define BLAS_zscal(n,alpha,Y,incy) \
{ \
    BLAS_INT N = n, INCY = incy ; \
    if (CHECK_BLAS_INT && !(EQ (N,n) && EQ (INCY,incy))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_ZSCAL (&N, MKCOMPLEX(alpha), MKCOMPLEX(Y), &INCY) ; \
    } \
}

#define BLAS_dger(m,n,alpha,X,incx,Y,incy,A,lda) \
{ \
    BLAS_INT M = m, N = n, LDA = lda, INCX = incx, INCY = incy ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (N,n) && EQ (LDA,lda) && \
          EQ (INCX,incx) && EQ (INCY,incy))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_DGER (&M, &N, alpha, X, &INCX, Y, &INCY, A, &LDA) ; \
    } \
}

#define BLAS_zgeru(m,n,alpha,X,incx,Y,incy,A,lda) \
{ \
    BLAS_INT M = m, N = n, LDA = lda, INCX = incx, INCY = incy ; \
    if (CHECK_BLAS_INT && !(EQ (M,m) && EQ (N,n) && EQ (LDA,lda) && \
          EQ (INCX,incx) && EQ (INCY,incy))) \
    { \
	BLAS_OK = FALSE ; \
    } \
    if (!CHECK_BLAS_INT || BLAS_OK) \
    { \
	BLAS_ZGER (&M, &N, alpha, MKCOMPLEX(X), &INCX, MKCOMPLEX(Y), &INCY, MKCOMPLEX(A), &LDA) ; \
    } \
}

#endif
