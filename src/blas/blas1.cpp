/** \file
 * \brief Implementation of some simple thread-parallel BLAS-1 operations
 * \author Aditya Kashi
 */

#include <cmath>
#include <limits>
#include "blas1.hpp"

namespace blasted {

template <typename scalar>
scalar maxnorm(const device_vector<scalar>& vec)
{
	scalar smax = std::numeric_limits<scalar>::min();
	const int N = static_cast<int>(vec.size());

#pragma omp parallel for default(shared) reduction(max:smax)
	for(int i = 0; i < N; i++)
		if(smax < std::abs(vec[i]))
			smax = std::abs(vec[i]);

	return smax;
}

template double maxnorm(const device_vector<double>& vec);

}
