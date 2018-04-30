/** \file srmatrixdefs.hpp
 * \brief Some definitions for everything depending on sparse-row storage
 * \author Aditya Kashi
 */

#include <limits>
#include <Eigen/Core>

#ifndef BLASTED_SRMATRIXDEFS_H
#define BLASTED_SRMATRIXDEFS_H

namespace blasted {

using Eigen::Dynamic;
using Eigen::RowMajor;
using Eigen::ColMajor;
using Eigen::StorageOptions;
using Eigen::Matrix;
template <typename scalar>
using Vector = Matrix<scalar,Dynamic,1>;

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
void destroyCRawBSRMatrix(CRawBSRMatrix<scalar,index>& rmat) {
	delete [] rmat.browptr;
	delete [] rmat.bcolind;
	delete [] rmat.vals;
	delete [] rmat.diagind;
}

template <typename scalar, typename index>
void destroyRawBSRMatrix(RawBSRMatrix<scalar,index>& rmat) {
	destroyCRawBSRMatrix(reinterpret_cast<CRawBSRMatrix<scalar,index>&>(rmat));
}

}
#endif
