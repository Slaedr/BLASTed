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

/// BLAS-2 operations for BSR matrices
template <typename mscalar, typename mindex, int bs, StorageOptions stor>
struct BLAS_BSR {
	/// The working (base) scalar type
	typedef typename std::remove_cv<mscalar>::type scalar;
	/// The working (base) index type
	typedef typename std::remove_cv<mindex>::type index;

	/// Matrix-vector product for BSR matrices
	static void matrix_apply(const SRMatrixStorage<mscalar,mindex>&& mat,
	                         const scalar *const xx, scalar *const __restrict yy);

	/// Computes z := a Ax + by for  scalars a and b and vectors x and y
	/**
	 * \param[in] mat The BSR matrix
	 * \warning xx must not alias zz.
	 */
	static void gemv3(const SRMatrixStorage<mscalar,mindex>&& mat,
	                  const scalar a, const scalar *const __restrict xx,
	                  const scalar b, const scalar *const yy, scalar *const zz);
};

/// BLAS-2 operations for CSR matrices
template <typename mscalar, typename mindex>
struct BLAS_CSR {
	/// The working (base) scalar type
	typedef typename std::remove_cv<mscalar>::type scalar;
	/// The working (base) index type
	typedef typename std::remove_cv<mindex>::type index;

	/// Matrix-vector product for CSR matrices
	static void matrix_apply(const SRMatrixStorage<mscalar,mindex>&& mat,
	                         const scalar *const xx, scalar *const __restrict yy);

	/// Computes z := a Ax + by for  scalars a and b and vectors x and y
	/**
	 * \param[in] mat The CSR matrix
	 * \warning xx must not alias zz.
	 */
	static void gemv3(const SRMatrixStorage<mscalar,mindex>&& mat,
	                  const scalar a, const scalar *const __restrict xx,
	                  const scalar b, const scalar *const yy, scalar *const zz);
};

/// GeMV for block compressed sparse column matrix
/** Computes z := a Ax + by for  scalars a and b and vectors x and y
 */
template <typename scalar, typename index, int bs, StorageOptions stor>
void bcsc_gemv3(const CRawBSCMatrix<const scalar, const index> *const mat,
                const scalar a, const scalar *const __restrict xx,
                const scalar b, const scalar *const yy, scalar *const zz);

}

#endif
