/** \file
 * \brief \ref ArrayList
 */

#ifndef BLASTED_ARRAY_LIST_H
#define BLASTED_ARRAY_LIST_H

#include <vector>
#include <utility>

namespace blasted { namespace internal {

/// A list of two-dimensional column-major arrays of different sizes
/** The aim is to have one long contiguous array for the entries of all the arrays, but retain
 * easy indexing into it.
 */
template <typename T>
class ColMajorArrayList
{
public:
	ArrayList();

	ArrayList(const int num_arrays, std::vector<int>&& pointers, std::vector<int>&& num_rows)
		: numarrays{num_arrays}, ptrs(std::move(pointers)), numrows(std::move(num_rows))
	{ }

	void create(const int num_arrays, std::vector<int>&& pointers, std::vector<int>&& num_rows)
	{
		numarrays = num_arrays;
		ptrs = std::move(pointers);
		numrows = std::move(num_rows);
	}

	T& operator(const int i_array, const int irow, const int jcol)
	{
		return vals[ptrs[i_array] + irow + jcol*numrows[i_array]];
	}

	const T& operator(const int i_array, const int irow, const int jcol) const
	{
		return vals[ptrs[i_array] + irow + jcol*numrows[i_array]];
	}

private:
	int numarrays;                   ///< Number of arrays
	std::vector<int> ptrs;           ///< Pointers to the beginning of each array
	std::vector<int> numrows;        ///< Number of rows in each array
	std::vector<T> vals;             ///< Array entries
};

}}

#endif
