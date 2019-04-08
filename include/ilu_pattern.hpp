/** \file
 * \brief Calculation of index lists needed for ILU
 * \author Aditya Kashi
 * 
 * This file is part of BLASTed.
 *   BLASTed is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   BLASTed is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with BLASTed.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BLASTED_ILU_PATTERN_H
#define BLASTED_ILU_PATTERN_H

#include <vector>
#include "bsr/srmatrixdefs.hpp"

namespace blasted {

/// Stores the locations that need to be processed for computing each entry of the ILU factorization
template<typename index>
struct ILUPositions
{
	/// Positions of entries in bcolind of the L matrix that are needed for a given entry of the ILU factors
	std::vector<index> lowerp;

	/// Positions in the bcolind array of the original matrix that need to be multiplied by corresponding
	///  entries of \ref lowerp.
	std::vector<index> upperp;

	/// Pointers into \ref lowerp and \ref upperp for the beginning of the lists corresponding to each
	///  non-zero entry
	std::vector<index> posptr;
};

/// Computes a list of positions in L and U that need to traversed for computing new ILU components
template <typename scalar, typename index>
ILUPositions<index> compute_ILU_positions_CSR_CSR(const CRawBSRMatrix<scalar,index> *const mat);

}

#endif
