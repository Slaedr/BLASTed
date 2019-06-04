/** \file
 * \brief Some processing needed for building SAI-type preconditioners
 */

#ifndef BLASTED_SAI_H
#define BLASTED_SAI_H

#include <vector>
#include "srmatrixdefs.hpp"

namespace blasted {

/// Stores indices of non-zeros in a CSR-type matrix corresponding to the local least-squares matrices
///  if a sparse approximate inverse preconditioner
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

/// Computes the pattern as described for \ref TriangularLeftSAIPattern
/** \param mat The input matrix for which the SAI pattern is to be determined. Note that the values of
 *             the non-zero entries are immaterial - only the sparsity pattern is required.
 *  \param fullsai Whether SAI (true) or incomplete SAI (false) preconditioner is desired.
 */
template <typename scalar, typename index>
TriangularLeftSAIPattern<index> compute_triangular_SAI_pattern(const CRawBSRMatrix<scalar,index>& mat,
                                                               const bool fullsai);

}

#endif
