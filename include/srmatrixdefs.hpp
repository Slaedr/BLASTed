/** \file srmatrixdefs.hpp
 * \brief Some definitions for everything depending on sparse-row storage
 * \author Aditya Kashi
 */

#ifndef BLASTED_SRMATRIXDEFS_H
#define BLASTED_SRMATRIXDEFS_H

#include <limits>
#include <Eigen/Core>

/// Cache line length in bytes, used for aligned allocation
#define CACHE_LINE_LEN 64

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::ColMajor;
using Eigen::StorageOptions;
using Eigen::Matrix;

/// Storage type for small segments of vectors
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

/// The storage type to use for each small dense block in case of BSR matrices
/** \warning Eigen assumes whatever alignment is necessary for efficient vectorization.
  *    This will be a problem when the data passed in by an external library is not aligned.
  *    If that is the case, we need to change the last template param to layout|Eigen::DontAlign.
  *    In that case, the code may be somewhat slower.
  */
template <typename scalar, int bs, Eigen::StorageOptions layout>
using Block_t = Eigen::Matrix<scalar,bs,bs,layout>;

/// Type to represent a small segment of a vector
/** Note that Eigen is asked to assume no alignment.
  */
template <typename scalar, int bs>
using Segment_t = Eigen::Matrix<scalar,bs,1,Eigen::ColMajor|Eigen::DontAlign>;

/// An (almost-)immutable compressed sparse block-row square matrix
/** The pointers and the number of (block-)rows are non-const to allow re-wrapping of another matrix.
 * Since objects of this type are used as members of other classes, we allow those classes to handle
 * mutability (or lack thereof) through the use of const member functions.
 */
template <typename scalar, typename index>
struct CRawBSRMatrix
{
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	const index *browptr;      ///< pointers to beginning block-rows as well as nnz blocks at the end
	const index *bcolind;      ///< block-column indices of non-zeros
	const scalar *vals;        ///< values of non-zero blocks
	const index *diagind;      ///< locations of the diagonal block in every block-row
	const index *browendptr;   ///< pointers to one-past-the-end of block-rows
	index nbrows;              ///< number of block rows

	/// Default constructor
	CRawBSRMatrix()
		: browptr{nullptr}, bcolind{nullptr}, vals{nullptr}, diagind{nullptr}, browendptr{nullptr},
		  nbrows{0}
	{ }

	/// Set data pointers
	CRawBSRMatrix(const index *const brptrs, const index *const bcinds,
	              const scalar *const values, const index *const diag_inds,
	              const index *const brendptrs, const index n_brows)
		: browptr{brptrs}, bcolind{bcinds}, vals{values}, diagind{diag_inds}, browendptr{brendptrs},
		  nbrows{n_brows}
	{ }
};

/// A compressed sparse block-row square matrix
template <typename scalar, typename index>
struct RawBSRMatrix
{
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	index *browptr;            ///< pointers to beginning block-rows, as well as nnz blocks at the end
	index *bcolind;            ///< block-column indices of non-zeros
	scalar *vals;              ///< values of non-zero blocks
	index *diagind;            ///< locations of the diagonal block in every block-row
	index *browendptr;         ///< pointers to one-past-the-end of block-rows
	index nbrows;              ///< number of block rows

	/// Default constructor
	RawBSRMatrix()
		: browptr{nullptr}, bcolind{nullptr}, vals{nullptr}, diagind{nullptr}, browendptr{nullptr},
		  nbrows{0}
	{ }

	/// Set data pointers
	RawBSRMatrix(index *const brptrs, index *const bcinds,
	             scalar *const values, index *const diag_inds,
	             index *const brendptrs, const index n_brows)
		: browptr{brptrs}, bcolind{bcinds}, vals{values}, diagind{diag_inds}, browendptr{brendptrs},
		  nbrows{n_brows}
	{ }
};

/// Frees all memory in case the memory was allocated with a simple new
template <typename scalar, typename index>
void destroyCRawBSRMatrix(CRawBSRMatrix<scalar,index>& rmat);

/// Frees all memory in case the memory was allocated with a simple new
template <typename scalar, typename index>
void destroyRawBSRMatrix(RawBSRMatrix<scalar,index>& rmat);

/// Frees all memory in case the memory was allocated with Boost align
template <typename scalar, typename index>
void alignedDestroyCRawBSRMatrix(CRawBSRMatrix<scalar,index>& rmat);

/// Frees all memory in case the memory was allocated with Boost align
template <typename scalar, typename index>
void alignedDestroyRawBSRMatrix(RawBSRMatrix<scalar,index>& rmat);

/// Allocates (aligned) memory for a new BSR matrix, copies a matrix into it and returns it
/** \warning ONLY for full matrices, NOT partial views.
 */
template <typename scalar, typename index, int bs>
RawBSRMatrix<scalar,index> copyRawBSRMatrix(const CRawBSRMatrix<scalar,index>& mat);

/// Returns a view to the lower triangular part of the matrix
/** Creates a new array for row pointers and row end pointers, but reuses values and column indices.
 * \warning Once done with the lower triangular matrix, ONLY browptr needs to be deleted.
 * \sa alignedDestroyCRawBSRMatrixTriangularView
 */
template <typename scalar, typename index>
CRawBSRMatrix<scalar,index> getLowerTriangularView(const CRawBSRMatrix<scalar,index>& mat);

/// Returns a view to the upper triangular part of the matrix
/** Creates a new array for row pointers and row-end pointers, but reuses values and column indices.
 * \warning Once done with the upper triangular matrix, ONLY browptr needs to be deleted.
 * \sa alignedDestroyCRawBSRMatrixTriangularView
 */
template <typename scalar, typename index>
CRawBSRMatrix<scalar,index> getUpperTriangularView(const CRawBSRMatrix<scalar,index>& mat);

/// Destroys a lower triangular view extracted using getLowerTriangularView or getUpperTriangularView
/** CRawBSRMatrix& will have to be reinterpret_cast to RawBSRMatrix&.
 */
template <typename scalar, typename index>
void alignedDestroyRawBSRMatrixTriangularView(CRawBSRMatrix<scalar,index>& mat);

}
#endif
