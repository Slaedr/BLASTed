/** \file
 * \brief Implementation of level-scheduling
 * \author Aditya Kashi
 */

#include <list>
#include "levelschedule.hpp"

namespace blasted {

template <typename scalar, typename index>
std::vector<index> computeLevels(const CRawBSRMatrix<scalar,index>& mat)
{
	std::vector<index> levels;

	// Build dependency lists
	std::vector<std::list<index>> depends(mat.nbrows);
	for(index irow = 0; irow < mat.nbrows; irow++)
	{
		for(index jj = mat.browptr[irow]; jj < mat.browptr[irow+1]; jj++)
			depends[irow].push_back(mat.bcolind[jj]);
	}

	index inode = 0;
	levels.push_back(0);
	index nlevels = 0;

	while(inode < mat.nbrows)
	{
		// 1. Find consecutive independent nodes

		while (depends[inode].first() >= inode && inode < mat.nbrows)
			inode++;

		levels.push_back(inode);
		nlevels++;

		// 2. remove dependency of remaining nodes on each node in this level

		// for each node in the new level..
		for(index jnode = levels[levels.size()-2]; jnode < inode; jnode++)
		{
			// go over the neighbors of the node..
			for(auto jnbr = depends[jnode].begin(); jnbr != depends[jnode].end(); jnbr++)
			{
				// remove jnode from the list of dependencies of the neighbor
				auto it = std::find(depends[*jnbr].begin(), depends[*jnbr].end(), jnode);

				if(it == depends[*jnbr].end())
					throw std::runtime_error("Faulty dependency list!");

				depends[*jnbr].erase(it);
			}
		}
	}

	assert(inode == mat.nbrows);

	return levels;
}

}
