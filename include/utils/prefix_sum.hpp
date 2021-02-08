
#ifndef BLASTED_PREFIX_SUM_H
#define BLASTED_PREFIX_SUM_H

namespace blasted {

/** \brief Parallel prefix sum
 * 
 * \param[in,out] counts  Array of to be summed.
 *     The last entry is never used, but is replaced by the sum of all entries.
 *     This function replaces the array by its prefix sum:
 *     The first entry is set to 0, and 
 *     for i > 0, count[i] = sum_{k=0}^{i-1} count[k].
 * \param[in] num_entries  The size of the array counts.
 */
template <typename index_type>
void prefix_sum(index_type *counts, int num_entries);

}

#endif
