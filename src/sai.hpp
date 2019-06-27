/** \file
 * \brief Some processing needed for building SAI-type preconditioners
 */

#ifndef BLASTED_SAI_H
#define BLASTED_SAI_H

#include <vector>
#include <boost/align/aligned_allocator.hpp>
#include "srmatrixdefs.hpp"

namespace blasted {

using boost::alignment::aligned_allocator;

/// The pattern required to assemble small least-squares problems for computing a left SAI preconditioner
/** The sparsity pattern of the SAI is set to that of A.
 */
template <typename index>
struct LeftSAIPattern
{
	/// Locations in the original matrix's bcolind array for each entry of all SAI LHS matrices
	std::vector<index,aligned_allocator<index,CACHE_LINE_LEN>> bpos;
	/// Row indices of entries in the SAI LHS matrix corresponding to \ref bpos
	std::vector<int,aligned_allocator<index,CACHE_LINE_LEN>> browind;
	/// Pointers to the start of every block-column of every SAI LHS matrix
	std::vector<index,aligned_allocator<index,CACHE_LINE_LEN>> bcolptr;
	/// Pointers into \ref bcolptr for the start of the LHS matrix of the least-squares problem
	///   for each block-row of the original matrix
	std::vector<index,aligned_allocator<index,CACHE_LINE_LEN>> sairowptr;

	/// Number of non-zero blocks in each block-row of the approximate inverse
	std::vector<int,aligned_allocator<int,CACHE_LINE_LEN>> nVars;
	/// Number of equations in the least-squares problem for each block-row of the approx inverse
	std::vector<int,aligned_allocator<int,CACHE_LINE_LEN>> nEqns;
};

template <typename scalar, typename index>
LeftSAIPattern<index> left_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat);

#if 0
/// Stores indices of non-zeros in a CSR-type matrix corresponding to the local least-squares matrices
///  of a sparse approximate inverse preconditioner
/** It is meant for use with a left approximate inverse preconditioner stored in a CSR-type format.
 */
template <typename index>
struct TriangularLeftSAIPattern
{
	/// For every row of lower approx inverse: non-zero location of L, stored as a column-major array,
	///  for each entry of the small LHS matrix
	std::vector<index> lowernz;
	/// Pointers into \ref lowernz for each row of M_l
	std::vector<index> ptrlower;
	/// For every row of upper approx inverse: non-zero location of U, stored as a column-major array,
	///  for each entry of the small LHS matrix
	std::vector<index> uppernz;
	/// Pointers into \ref uppernz for each row of M_l
	std::vector<index> ptrupper;
	/// Number of columns of the original lower matrix required for each row of M
	/** Each relevant column of A becomes a row of the local least-squares LHS.
	 */
	std::vector<int> lowerMConstraints;
	/// Number of columns of the original upper matrix required for each row of M
	std::vector<int> upperMConstraints;
};

/// Computes the pattern (as described for \ref TriangularLeftSAIPattern) of a SAI(1,2) preconditioner
///  for the lower and upper triangular parts (both including the diagonal) of a CSR-type matrix
/** Note that SAI(k,l) means that for each point in the grid, k layers of neighbours are used for the
 * sparsity pattern of the row of the inverse, while l layers of neighbours are used to provide
 * constraints for the determination of those non-zeros.
 * \param mat The input matrix for which the SAI pattern is to be determined. Note that the values of
 *             the non-zero entries are immaterial - only the sparsity pattern is required.
 */
template <typename scalar, typename index>
TriangularLeftSAIPattern<index> triangular_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat);

/// Computes the pattern (as described for \ref TriangularLeftSAIPattern) of a ISAI(1,1) preconditioner
///  for the lower and upper triangular parts (both including the diagonal) of a CSR-type matrix
/** \param mat The input matrix for which the SAI pattern is to be determined. Note that the values of
 *              the non-zero entries are immaterial - only the sparsity pattern is required.
 */
template <typename scalar, typename index>
TriangularLeftSAIPattern<index> triangular_incomp_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat);
#endif

}

#endif
