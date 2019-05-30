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

	/// Access lists of rows having non-zeros in any column
	const std::vector<index>& col_nonzero_rows() const { return col_rows; }

	/// Access locations, in the original BSR matrix, of non-zeros of each column
	const std::vector<index>& col_nonzero_locations() const { return rows_loc; }

	/// Pointers into the lists of non-zeros where each column's list begins
	/** Use to access data from \ref col_nonzero_rows and \ref col_nonzero_locations
	 */
	const std::vector<index>& col_pointers() const { return ptrs; }

protected:
	/// The indices of rows that contain a non-zero for each column
	std::vector<index> col_rows;
	/// Locations of the non-zero in each row corresponding to each column
	/** Ordered the same way as \ref rows_nz
	 */
	std::vector<index> rows_loc;
	/// Pointers into \ref rows where the list of each column begins
	std::vector<index> ptrs;
};

}

#endif
