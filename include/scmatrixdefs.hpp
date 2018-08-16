/** \file scmatrixdefs.hpp
 * \brief Some definitions for everything depending on sparse-column storage
 * \author Aditya Kashi
 */

#ifndef BLASTED_SCMATRIXDEFS_H
#define BLASTED_SCMATRIXDEFS_H

#include <limits>
#include "srmatrixdefs.hpp"

namespace blasted {

/// A compressed sparse block-column square matrix
template <typename scalar, typename index>
struct RawBSCMatrix
{
	static_assert(std::numeric_limits<index>::is_integer, "Integer index type required!");
	static_assert(std::numeric_limits<index>::is_signed, "Signed index type required!");

	/// Array of pointers into \ref browind that point to first entries of (block-)columns
	index * bcolptr;
	/// Array of row indices of each non-zero entry, stored in the same order as \ref vals
	index * browind;
	/// Array of non-zero values
	scalar * vals;
	/// Pointers into \ref browind pointing to diagonal entries in each column
	index * diagind;
	/// Number of (block-)columns
	index nbcols;
};

/// Converts a (block-) sparse-row matrix to a (block-) sparse-column matrix
/** Assumes a square matrix.
 */
template <typename scalar, typename index, int bs>
RawBSCMatrix<scalar,index> convert_BSR_to_BSC(const CRawBSRMatrix<scalar,index> *const rmat);

/// Frees storage
template <typename scalar, typename index>
void destroyRawBSCMatrix(RawBSCMatrix<scalar,index>& mat);

}
#endif
