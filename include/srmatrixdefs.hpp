/** \file srmatrixdefs.hpp
 * \brief Some definitions for everything depending on sparse-row storage
 * \author Aditya Kashi
 */

#ifndef BLASTED_SRMATRIXDEFS_H
#define BLASTED_SRMATRIXDEFS_H

#include <limits>
#include "blasted_config.hpp"
#include "arrayview.hpp"

namespace blasted {

template <typename scalar, typename index>
struct SRMatrixStorage;

/// Moves the argument into the immutable return type (the argument is then nulled)
/** This is useful for 'casting' SRMatrixStorage<double,int> into
 * SRMatrixStorage<const double, const int>, for example.
 */
template <typename scalar, typename index>
SRMatrixStorage<typename std::add_const<scalar>::type, typename std::add_const<index>::type>
move_to_const(SRMatrixStorage<scalar,index>&& smat);

/// Sparse-row type storage for matrix
template <typename scalar, typename index>
struct SRMatrixStorage
{
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");

	ArrayView<index> browptr;            ///< pointers to beginning block-rows
	ArrayView<index> bcolind;            ///< block-column indices of non-zeros
	ArrayView<scalar> vals;              ///< values of non-zero blocks
	ArrayView<index> diagind;            ///< locations of the diagonal block in every block-row
	ArrayView<index> browendptr;         ///< pointers to one-past-the-end of block-rows
	index nbrows;                        ///< number of block rows
	index nnzb;                          ///< number of non-zero blocks
	index nbstored;               ///< total number of non-zero blocks actually stored in the array vals

	/// Default constructor - trivially sets as a zero matrix with no allocation
	SRMatrixStorage();

	/// Wrap data pointers
	SRMatrixStorage(index *const brptrs, index *const bcinds,
	                scalar *const values, index *const diag_inds, index *const brendptrs,
	                const index n_brows, const index n_nzb, const index n_bstored);

	/// Move arrays into this object
	SRMatrixStorage(ArrayView<index>&& brptrs, ArrayView<index>&& bcinds, ArrayView<scalar>&& vals,
	                ArrayView<index>&& diag_inds, ArrayView<index>&& brendptrs,
	                const index n_brows, const index n_nzb, const index n_bstored);

	/// Move constructor from another SRMatrixStorage
	SRMatrixStorage(SRMatrixStorage<scalar,index>&& other);

	friend
	SRMatrixStorage<typename std::add_const<scalar>::type, typename std::add_const<index>::type>
	move_to_const<>(SRMatrixStorage<scalar,index>&& smat);
};

/// Wraps the arrays of the argument in an immutable matrix, while keeping the argument unchanged
/** After this function returns, the returned matrix and the argument matrix refer to the same CSR
 * arrays.
 */
template <typename scalar, typename index>
SRMatrixStorage<typename std::add_const<scalar>::type, typename std::add_const<index>::type>
share_with_const(const SRMatrixStorage<scalar,index>& smat);

/// An (almost-)immutable sparse block-row square matrix
/** The pointers and the number of (block-)rows are non-const to allow re-wrapping of another matrix.
 * Since objects of this type are used as members of other classes, we allow those classes to handle
 * mutability (or lack thereof) through the use of const member functions.
 *
 * Note that the 'actual' or logical matrix may not be the same as what is stored. This is because
 * pointers denoting ends of rows need not be the same as pointers denoting the start of the following
 * rows.
 */
template <typename scalar, typename index>
struct CRawBSRMatrix
{
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	const index *browptr;      ///< pointers to beginning block-rows
	const index *bcolind;      ///< block-column indices of non-zeros
	const scalar *vals;        ///< values of non-zero blocks
	const index *diagind;      ///< locations of the diagonal block in every block-row
	const index *browendptr;   ///< pointers to one-past-the-end of block-rows
	index nbrows;              ///< number of block rows
	index nnzb;                ///< total number of non-zero blocks in the matrix
	index nbstored;            ///< total number of non-zero blocks actually stored in \ref vals

	/// Default constructor
	CRawBSRMatrix()
		: browptr{nullptr}, bcolind{nullptr}, vals{nullptr}, diagind{nullptr}, browendptr{nullptr},
		  nbrows{0}, nnzb{0}, nbstored{0}
	{ }

	/// Set data pointers
	CRawBSRMatrix(const index *const brptrs, const index *const bcinds,
	              const scalar *const values, const index *const diag_inds, const index *const brendptrs,
	              const index n_brows, const index n_nzb, const index n_bstored)
		: browptr{brptrs}, bcolind{bcinds}, vals{values}, diagind{diag_inds}, browendptr{brendptrs},
		  nbrows{n_brows}, nnzb{n_nzb}, nbstored{n_bstored}
	{ }
};

/// A sparse block-row square matrix
/** This is a mutable version of \ref CRawBSRMatrix and is supposed to be byte-equivalent to it.
 * This allows safely reinterpreting a RawBSRMatrix as a CRawBSRMatrix.
 * Note that CRawBSRMatrix would be unnecessary if we used a vector-like container for the member arrays.
 */
template <typename scalar, typename index>
struct RawBSRMatrix
{
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	index *browptr;            ///< pointers to beginning block-rows
	index *bcolind;            ///< block-column indices of non-zeros
	scalar *vals;              ///< values of non-zero blocks
	index *diagind;            ///< locations of the diagonal block in every block-row
	index *browendptr;         ///< pointers to one-past-the-end of block-rows
	index nbrows;              ///< number of block rows
	index nnzb;                ///< number of non-zero blocks
	index nbstored;            ///< total number of non-zero blocks actually stored in the array vals

	/// Default constructor
	RawBSRMatrix()
		: browptr{nullptr}, bcolind{nullptr}, vals{nullptr}, diagind{nullptr}, browendptr{nullptr},
		  nbrows{0}, nnzb{0}
	{ }

	/// Set data pointers
	RawBSRMatrix(index *const brptrs, index *const bcinds,
	             scalar *const values, index *const diag_inds,
	             index *const brendptrs, const index n_brows, const index n_nzb, const index n_bstored)
		: browptr{brptrs}, bcolind{bcinds}, vals{values}, diagind{diag_inds}, browendptr{brendptrs},
		  nbrows{n_brows}, nnzb{n_nzb}, nbstored{n_bstored}
	{ }
};

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
void alignedDestroyRawBSRMatrixTriangularView(RawBSRMatrix<scalar,index>& mat);

}
#endif
