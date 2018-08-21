/** \file
 * \brief Some discrete algorithms requried for re-ordering etc
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

#ifndef BLASTED_HELPER_ALGORITHMS_H
#define BLASTED_HELPER_ALGORITHMS_H

namespace blasted {

/// Functionality only used in implementation(s); not part of the public interface of the library
namespace internal {

/// Sorts two "corresponding" arrays according to the first array
/** This is meant for sorting the array of 'inner' indices and the array of non-zero values for
 * on particular outer index. This means, for example, sorting by columns within a row of a sparse-
 * row matrix. The non-zero values are sorted accordingly. The arrays are sorted in place.
 * \param N Size of the index array to be sorted. See below for more details.
 * \param colind The array of inner indices. This has length N.
 * \param vals The array of non-zero values corresponding to inner indices in \ref colind.
 *   However, each inner index is assumed to be associated with a bs X bs block of non-zeros. Thus,
 *   the length of vals is N*bs*bs.
 */
template <typename scalar, typename index, int bs>
void sortBlockInnerDimension(const index N, index *const colind, scalar *const vals);

}
}

#endif
