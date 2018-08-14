/** \file
 * \brief Some schemes to reorder and scale matrices for various purposes
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

#ifndef BLASTED_REORDERING_SCALING_H
#define BLASTED_REORDERING_SCALING_H

#include <vector>
#include "srmatrixdefs.hpp"

namespace blasted {

/// For a reordering or scaling, whether to apply it or its inverse
enum RSApplyMode {FORWARD, INVERSE};
/// For a reordering or scaling, whether to apply it to rows or columns of a matrix, or both
enum RSApplyDir {ROW, COLUMN, BOTH};

/// Abstract handler for computing a reordering of a matrix stored in a sparse-row format
template <typename scalar, typename index>
class Reordering
{
public:
	/// Do-nothing constructor
	Reordering();

	/// Destructor
	virtual ~Reordering();

	/// Set an ordering from a permutation vector
	/** \param rord Row ordering vector (can be nullptr, in which case it's ignored)
	 * \param cord Column ordering vector (can be nullptr, in which case it's ignored)
	 * \param length Length of the vector (dimension of the vector space)
	 */
	void setOrdering(const index *const rord, const index *const cord, const index length);

	/// Compute an ordering
	/** \param mat The matrix on which some algorithm is applied to compute the ordering
	 */
	virtual void compute(const CRawBSRMatrix<scalar,index>& mat) = 0;

	/// Apply the ordering to a matrix
	/** Either row or column reordering (or both) is done depending on the specific reordering method.
	 */
	virtual void applyOrdering(RawBSRMatrix<scalar,index>& mat) const = 0;

	/// Apply the ordering to a vector
	/** \param vec The vector to reorder
	 * \param mode Whether to apply the computed ordering or its inverse
	 * \param dir Whether to apply the row ordering or the column ordering
	 */
	virtual void applyOrdering(scalar *const vec,
	                           const RSApplyMode mode, const RSApplyDir dir) const = 0;

protected:

	/// Row permutation vector
	std::vector<index> rp;
	/// Column permutation vector
	std::vector<index> cp;
};

/// Abstract handler for computing a reordering and a scaling of a matrix stored in sparse-row format
/** Reordering::compute should also computes the scaling in this case.
 */
template <typename scalar, typename index>
class ReorderingScaling : public Reordering<scalar,index>
{
public:
	/// Do-nothing constructor
	ReorderingScaling();

	/// Apply only scaling to a matrix
	virtual void applyScaling(RawBSRMatrix<scalar,index>& mat) = 0;

	/// Apply only scaling to a vector
	/** \param vec The vector to scale
	 * \param mode Whether to apply the scaling or its inverse
	 * \param dir Whether to apply the row scaling or the column scaling
	 */
	virtual void applyScaling(scalar *const vec
	                          const RSApplyMode mode, const RSApplyDir dir) = 0;

protected:
	using Reordering<scalar,index>::rp;
	using Reordering<scalar,index>::cp;
};

}

#endif
