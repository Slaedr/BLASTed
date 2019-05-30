/** \file
 * \brief Adjacency lists of DOFs of a problem
 */

#ifndef BLASTED_ADJACENCY_LISTS_H
#define BLASTED_ADJACENCY_LISTS_H

#include <vector>
#include "srmatrixdefs.hpp"

namespace blasted {

/// Column-oriented adjacency information for a sparse-row matrix
template <typename scalar, typename index>
class ColumnAdjacency
{
public:
	/// Computes the adjacency lists of a CSR-type matrix
	ColumnAdjacency(const CRawBSRMatrix<scalar,index>& mat);

	/// Access lists of neighbours of any vertex \sa alists
	const std::vector<index>& nonzero_rows() const { return rows_nz; }

	/// Pointers into the list of non-zero rows \ref rows_nz, where each column's list begins
	const std::vector<index>& vertex_pointers() const { return ptrs; }

protected:
	/// The indices of rows that contain a non-zero for each column
	std::vector<index> rows_nz;
	/// Locations of non-zeros in the rows corresponding to each column
	std::vector<index> rows_loc;
	/// Pointers into \ref rows where the list of each column begins
	std::vector<index> ptrs;
};

}

#endif
