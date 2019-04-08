/** \file
 * \brief Implementation of level-scheduling
 * \author Aditya Kashi
 */

#include <list>
#include "bsr/levelschedule.hpp"

namespace blasted {

template <typename scalar, typename index>
std::vector<index> computeLevels(const CRawBSRMatrix<scalar,index> *const mat)
{
	std::vector<index> levels;

	// Build dependency lists
	std::vector<std::list<index>> depends(mat->nbrows);
	for(index irow = 0; irow < mat->nbrows; irow++)
	{
		for(index jj = mat->browptr[irow]; jj < mat->browptr[irow+1]; jj++)
			depends[irow].push_back(mat->bcolind[jj]);
	}

	index inode = 0;
	levels.push_back(0);
	index nlevels = 0;

	while(inode < mat->nbrows)
	{
		// 1. Find consecutive independent nodes

		while (inode < mat->nbrows && depends[inode].front() >= inode) {
			inode++;
		}

		// and add those nodes (ie., add the end of the consecutive set of nodes) to this level.
		levels.push_back(inode);
		nlevels++;

		// 2. Remove dependency of remaining nodes on each node in this level

		// for each node in the new level..
		for(index jnode = levels[levels.size()-2]; jnode < inode; jnode++)
		{
			// go over the neighbors of the node (because the sparsity structure is assumed symmetric)
			for(auto jnbr = depends[jnode].begin(); jnbr != depends[jnode].end(); jnbr++)
			{
				// Every node is its own dependency - skip that
				if(*jnbr == jnode)
					continue;

				// find jnode in the list of dependencies of the neighbor..
				auto it = std::find(depends[*jnbr].begin(), depends[*jnbr].end(), jnode);

				// (jnode must be found because the sparsity structure is symmetric)
				if(it == depends[*jnbr].end())
					throw std::runtime_error("Faulty dependency list!");

				// .. and remove it.
				depends[*jnbr].erase(it);
			}
		}
	}

	assert(inode == mat->nbrows);
	assert(mat->nbrows == levels.back());
	assert(nlevels+1 == static_cast<index>(levels.size()));
	printf(" LevelSchedule: Found %d levels.\n", nlevels);

	return levels;
}

template std::vector<int> computeLevels(const CRawBSRMatrix<double,int> *const mat);

}
