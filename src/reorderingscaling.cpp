/** \file
 * \brief Implementation of schemes to reorder and scale matrices for various purposes
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

#include "reorderingscaling.hpp"

namespace blasted {

template <typename scalar, typename index>
Reordering<scalar,index>::Reordering()
{ }

template <typename scalar, typename index>
Reordering<scalar,index>::~Reordering()
{ }

template <typename scalar, typename index>
Reordering<scalar,index>::setOrdering(const index *const rord, const index *const cord,
                                      const index length)
{
	if(rord) {
		rp.resize(length);
		for(index i = 0; i < length; i++)
			rp[i] = rord[i];
	}
	if(cord) {
		cp.resize(length);
		for(index i = 0; i < length; i++)
			cp[i] = cord[i];
	}
}

}
