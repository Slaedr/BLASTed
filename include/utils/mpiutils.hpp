/** \file
 * \brief Some convenience functions related to MPI
 * \author Aditya Kashi
 */

#ifndef BLASTED_MPIUTILS_H
#define BLASTED_MPIUTILS_H

#include <mpi.h>
#include <stdexcept>

namespace blasted {

inline int get_mpi_size(MPI_Comm comm)
{
	int size;
	int ierr = MPI_Comm_size(comm, &size);
	if(ierr)
		throw std::runtime_error("Could not get MPI size!");
	return size;
}

inline int get_mpi_rank(const MPI_Comm comm)
{
	int rank;
	int ierr = MPI_Comm_rank(comm, &rank);
	if(ierr)
		throw std::runtime_error("Could not get MPI rank!");
	return rank;
}

}

#endif
