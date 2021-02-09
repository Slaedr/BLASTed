
#include <omp.h>
#include <algorithm>
#include <cmath>

namespace blasted {


template <typename T>
static inline T power(const T x, const int exponent)
{
    T ans = static_cast<T>(1);
    for (int i = 0; i < exponent; i++) {
        ans *= x;
    }
    return ans;
}


template <typename IndexType>
void prefix_sum(IndexType *const counts, const int num_entries)
{
    if (num_entries <= 1) {
		if(num_entries > 0)
			counts[0] = 0;
        return;
    }

    const int nthreads = omp_get_max_threads();
    const IndexType def_num_witems = (num_entries - 1) / nthreads + 1;

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const IndexType startidx = thread_id * def_num_witems;
        const IndexType endidx =
            std::min(num_entries, (thread_id + 1) * def_num_witems);
        const IndexType startval = counts[startidx];

#pragma omp barrier

        IndexType partial_sum = startval;
        for (IndexType i = startidx + 1; i < endidx; ++i) {
            auto nnz = counts[i];
            counts[i] = partial_sum;
            partial_sum += nnz;
        }
        if (thread_id != nthreads - 1) {
            counts[endidx] = partial_sum;
        }
    }

    counts[0] = 0;

    const auto levels = static_cast<int>(std::ceil(std::log(nthreads)));
    for (int ilvl = 0; ilvl < levels; ilvl++) {
        const IndexType factor = power(2, (ilvl + 1));
        const IndexType lvl_num_witems = factor * def_num_witems;
        const int ntasks = (nthreads - 1) / factor + 1;

#pragma omp parallel for
        for (int itask = 0; itask < ntasks; itask++) {
            const IndexType startidx = std::min(
                num_entries, lvl_num_witems / 2 + itask * lvl_num_witems + 1);
            const IndexType endidx =
                std::min(num_entries, (itask + 1) * lvl_num_witems + 1);
            const IndexType baseval = counts[startidx - 1];
            for (int i = startidx; i < endidx; i++) {
                counts[i] += baseval;
            }
        }
    }
}

template
void prefix_sum(int *const counts, const int num_entries);

}
