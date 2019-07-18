/** \file
 * \brief Functions for matrix-vector products etc
 * \author Aditya Kashi
 */

#ifndef BLASTED_MATVECS_H
#define BLASTED_MATVECS_H

#include <Eigen/Core>
#include "srmatrixdefs.hpp"
#include "scmatrixdefs.hpp"

namespace blasted {

/// Matrix-vector product for BSR matrices
template <typename scalar, typename index, int bs, StorageOptions stor>
void bsr_matrix_apply(const SRMatrixStorage<const scalar, const index> *const mat,
                      const scalar *const xx, scalar *const __restrict yy);

/// Computes z := a Ax + by for  scalars a and b and vectors x and y
/**
 * \param[in] mat The BSR matrix
 * \warning xx must not alias zz.
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void bsr_gemv3(const SRMatrixStorage<const scalar, const index> *const mat,
               const scalar a, const scalar *const __restrict xx,
               const scalar b, const scalar *const yy, scalar *const zz);

/// Matrix-vector product for CSR matrices
template <typename scalar, typename index>
void csr_matrix_apply(const SRMatrixStorage<const scalar, const index> *const mat,
                      const scalar *const xx, scalar *const __restrict yy);

/// Computes z := a Ax + by for CSR matrix A, scalars a and b and vectors x and y
template <typename scalar, typename index>
void csr_gemv3(const SRMatrixStorage<const scalar, const index> *const mat,
               const scalar a, const scalar *const __restrict xx,
               const scalar b, const scalar *const yy, scalar *const zz);

/// GeMV for block compressed sparse column matrix
/** Computes z := a Ax + by for  scalars a and b and vectors x and y
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void bcsc_gemv3(const CRawBSCMatrix<const scalar, const index> *const mat,
                const scalar a, const scalar *const __restrict xx,
                const scalar b, const scalar *const yy, scalar *const zz);

}

#endif
