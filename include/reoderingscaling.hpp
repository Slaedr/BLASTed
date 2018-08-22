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

/// Handler for computing a reordering of a matrix stored in a sparse-(block-)row format
/** The ordering of entries within a small dense block is not altered.
 */
template <typename scalar, typename index, int bs>
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
	 * \param length Length of the ordering vectors
	 */
	void setOrdering(const index *const rord, const index *const cord, const index length);

	/// Compute an ordering
	/** \param mat The matrix on which some algorithm is applied to compute the ordering
	 *
	 * Implementations should leave the size of \ref rp as zero if a row-reordering is not to be done
	 * (rather than explicitly setting it to identity) for efficiency, and similary for \ref cp if
	 * column reordering is not to be done.
	 */
	virtual void compute(const CRawBSRMatrix<scalar,index>& mat) = 0;

	/// Apply the ordering to a matrix
	/** Either row or column reordering (or both) is done depending on the specific reordering method.
	 * Always applies the forward (original) ordering.
	 */
	virtual void applyOrdering(RawBSRMatrix<scalar,index>& mat) const;

	/// Apply the ordering (or its inverse) to a vector
	/** \param vec The vector to reorder
	 * \param mode Whether to apply the computed ordering or its inverse
	 * \param dir Whether to apply the row ordering or the column ordering
	 */
	virtual void applyOrdering(scalar *const vec,
	                           const RSApplyMode mode, const RSApplyDir dir) const;

protected:

	/// Row permutation vector
	std::vector<index> rp;
	/// Column permutation vector
	std::vector<index> cp;
};

/// Abstract handler for computing a reordering and a scaling of a matrix stored in sparse-row format
/** Reordering::compute should also computes the scaling in this case.
 */
template <typename scalar, typename index, int bs>
class ReorderingScaling : public Reordering<scalar,index,bs>
{
public:
	/// Do-nothing constructor
	ReorderingScaling();

	/// Apply only scaling to a matrix
	virtual void applyScaling(RawBSRMatrix<scalar,index>& mat) const;

	/// Apply only scaling to a vector
	/** \param vec The vector to scale
	 * \param mode Whether to apply the scaling or its inverse
	 * \param dir Whether to apply the row scaling or the column scaling
	 */
	virtual void applyScaling(scalar *const vec,
	                          const RSApplyMode mode, const RSApplyDir dir) const;

protected:
	using Reordering<scalar,index,bs>::rp;
	using Reordering<scalar,index,bs>::cp;

	/// Row scaling vector
	std::vector<scalar> rowscale;
	/// Column scaling vector
	std::vector<scalar> colscale;
};

}

#endif
