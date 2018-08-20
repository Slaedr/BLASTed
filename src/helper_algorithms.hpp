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
template <typename scalar, typename index, int bs>
void sortBlockInnerDimension(index *const colind, scalar *const vals);

}
}

#endif
