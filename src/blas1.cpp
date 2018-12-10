/** \file
 * \brief Implementation of some simple thread-parallel BLAS-1 operations
 * \author Aditya Kashi
 */

#include <cmath>
#include <limits>
#include "blas1.hpp"

namespace blasted {

template <typename scalar, typename index>
index maxnorm(const index N, const scalar *const vec)
{
	scalar smax = std::numeric_limits<scalar>::min();
#pragma omp parallel for default(shared) reduction(max:smax)
	for(index i = 0; i < N; i++)
		if(smax < std::abs(vec[i]))
			smax = std::abs(vec[i]);

	return smax;
}

template int maxnorm<double,int>(const int N, const double *const vec);

}
