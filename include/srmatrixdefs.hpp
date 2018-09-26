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
	const index *browptr;
	const index *bcolind;
	const scalar *vals;
	const index *diagind;
	index nbrows;
};

/// A compressed sparse block-row square matrix
template <typename scalar, typename index>
struct RawBSRMatrix
{
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");
	index * browptr;
	index * bcolind;
	scalar * vals;
	index * diagind;
	index nbrows;
};

template <typename scalar, typename index>
void destroyCRawBSRMatrix(CRawBSRMatrix<scalar,index>& rmat);

template <typename scalar, typename index>
void destroyRawBSRMatrix(RawBSRMatrix<scalar,index>& rmat);

template <typename scalar, typename index>
void alignedDestroyCRawBSRMatrix(CRawBSRMatrix<scalar,index>& rmat);

template <typename scalar, typename index>
void alignedDestroyRawBSRMatrix(RawBSRMatrix<scalar,index>& rmat);

/// Allocates (aligned) memory for a new BSR matrix, copies a matrix into it and returns it
template <typename scalar, typename index, int bs>
RawBSRMatrix<scalar,index> copyRawBSRMatrix(const CRawBSRMatrix<scalar,index>& mat);

}
#endif
