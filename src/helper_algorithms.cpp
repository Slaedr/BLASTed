/** \file
 * \brief Implementation of some discrete algorithms 
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

#include <algorithm>
#include <boost/align/aligned_allocator.hpp>
#include "blasted_config.hpp"
#include "helper_algorithms.hpp"

namespace blasted {
namespace internal {

/** Currently, this uses a simple O(N^2) algorithm because we don't expect one row or column
 * to have a lot of non-zeros. In fact, for PDEs on grids, we expect N = O(1).
 */
template <typename scalar, typename index, int bs>
void sortBlockInnerDimension(const index N, index *const colind, scalar *const vals)
{
	for(index i = 0; i < N-1; i++) {

		// find the max inner index (eg. column index) in the outer index entity (eg. row)
		index *const it = std::max_element(colind,colind+N-i);
		// find position of the max index in this outer index entity (eg. row)
		const ptrdiff_t max_pos = it - colind;

		// swap the max and the last
		const index tempi = colind[N-1-i];
		colind[N-1-i] = *it;
		*it = tempi;
		// swap non-zero blocks
		scalar tempv[bs*bs];
		for(int k = 0; k < bs*bs; k++)
			tempv[k] = vals[(N-1-i)*bs*bs + k];
		for(int k = 0; k < bs*bs; k++) {
			vals[(N-1-i)*bs*bs + k] = vals[max_pos*bs*bs + k];
			vals[max_pos*bs*bs + k] = tempv[k];
		}
	}
}

// for testing
template void sortBlockInnerDimension<double,int,1>(const int N,
                                                    int *const colind, double *const vals);
template void sortBlockInnerDimension<double,int,2>(const int N,
                                                    int *const colind, double *const vals);
template void sortBlockInnerDimension<double,int,4>(const int N,
                                                    int *const colind, double *const vals);
template void sortBlockInnerDimension<double,int,5>(const int N,
                                                    int *const colind, double *const vals);
template void sortBlockInnerDimension<double,int,7>(const int N,
                                                    int *const colind, double *const vals);

template <typename index>
void inclusive_scan(std::vector<index>& v)
{
	// serial
	for(size_t i = 1; i < v.size(); i++)
		v[i] += v[i-1];
}

template void inclusive_scan(std::vector<int>& v);

// template <typename index, typename allocator>
// void inclusive_scan(std::vector<index,allocator>& v)
// {
// 	// serial
// 	for(size_t i = 1; i < v.size(); i++)
// 		v[i] += v[i-1];
// }

// template void inclusive_scan(std::vector<int,boost::alignment::aligned_allocator<int,CACHE_LINE_LEN>>& v);

template <typename index>
void inclusive_scan(device_vector<index>& v)
{
	// serial
	for(size_t i = 1; i < v.size(); i++)
		v[i] += v[i-1];
}

template void inclusive_scan(device_vector<int>& v);

template <typename index>
std::vector<index> inclusive_scan(const std::vector<index>& v)
{
	std::vector<index> scanned(v);
	inclusive_scan(scanned);
	return scanned;
}

template std::vector<int> inclusive_scan(const std::vector<int>& v);

}
}
