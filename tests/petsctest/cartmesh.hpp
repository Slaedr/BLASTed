/** \file cartmesh.hpp
 * \brief Specification of single-block but distributed, non-uniform Cartesian grid using PETSc
 * \author Aditya Kashi
 */

#ifndef CARTMESH_H
#define CARTMESH_H

#include <petscdm.h>
#include <petscdmda.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NDIM 3

#define NSTENCIL 7

#define PI 3.141592653589793238

/// Non-uniform Cartesian grid
/** 
 * We store the on-dimensional locations of points along 3 orthogonal vectors whose
 * tensor product defines the grid. So, we store only 
 * - the x-coordinates of points lying along the "back lower horizontal" line of the cube
 *     containing the mesh
 * - the y-coordinates of points lying along the "back upper vertical" line of the cube
 * - the z-coordinates of points lying along the line "coming out" of the "bottom-left" corner 
 *     of the "back" plane
 */
class CartMesh
{
protected:
	/// Array storing the number of points on each coordinate axis
	PetscInt npoind[NDIM];

	/// Stores an array for each of the 3 axes 
	/// coords[i][j] refers to the j-th node along the i-axis
	PetscReal ** coords;			
	
	PetscInt npointotal;			///< Total number of points in the grid
	PetscInt ninpoin;				///< Number of internal (non-boundary) points
	PetscReal h;					///< Mesh size parameter

	// Stuff related to multiprocess
	PetscMPIInt nprocs[NDIM];		///< Number of processors in each dimension
	PetscMPIInt ntprocs;			///< Total number of processors

	/// Computes the mesh size parameter h
	/** Sets h as the length of the longest diagonal of all cells.
	 */
	void computeMeshSize();

public:
	CartMesh(const PetscInt npdim[NDIM], const PetscInt num_partitions);

	CartMesh(const PetscInt npdim[NDIM], PetscInt ndofpernode, PetscInt stencil_width,
		DMBoundaryType bx, DMBoundaryType by, DMBoundaryType bz, DMDAStencilType stencil_type, 
		DM *const dap, PetscMPIInt rank);

	~CartMesh();

	/// Returns the number of points along a coordinate direction
	PetscInt gnpoind(const int idim) const
	{
#if DEBUG == 1
		if(idim >= NDIM) {
			std::printf("! Cartmesh: gnpoind(): Invalid dimension %d!\n", idim);
			return 0;
		}
#endif
		return npoind[idim];
	}

	/// Returns a coordinate of a grid point
	/** \param[in] idim The coordinate line along which the point to be queried lies
	 * \param[in] ipoin Index of the required point in the direction idim
	 */
	PetscReal gcoords(const int idim, const PetscInt ipoin) const
	{
#if DEBUG == 1
		if(idim >= NDIM) 
		{
			std::printf("! Cartmesh: gcoords(): Invalid dimension!\n");
			return 0;
		}
		if(ipoin >= npoind[idim]) 
		{
			std::printf("! Cartmesh: gcoords(): Point does not exist!\n");
			return 0;
		}
#endif
		return coords[idim][ipoin];
	}

	PetscInt gnpointotal() const { return npointotal; }
	PetscInt gninpoin() const { return ninpoin; }
	PetscReal gh() const { return h; }

	const PetscInt * pointer_npoind() const
	{
		return npoind;
	}

	const PetscReal *const * pointer_coords() const
	{
		return coords;
	}

	/// Generate a non-uniform mesh in a cuboid corresponding to Chebyshev points in each direction
	/** For interval [a,b], a Chebyshev distribution of N points including a and b is computed as
	 * x_i = (a+b)/2 + (a-b)/2 * cos(pi - i*theta)
	 * where theta = pi/(N-1)
	 *
	 * \param[in] rmin Array containing the lower bounds of the domain along each coordinate
	 * \param[in] rmax Array containing the upper bounds of the domain along each coordinate
	 * \param[in] rank The MPI rank of the current process
	 */
	void generateMesh_ChebyshevDistribution(const PetscReal rmin[NDIM], const PetscReal rmax[NDIM], 
			const PetscMPIInt rank);
	
	/// Generates grid with uniform spacing
	/**
	 * \param[in] rmin Array containing the lower bounds of the domain along each coordinate
	 * \param[in] rmax Array containing the upper bounds of the domain along each coordinate
	 * \param[in] rank The MPI rank of the current process
	 */
	void generateMesh_UniformDistribution(const PetscReal rmin[NDIM], const PetscReal rmax[NDIM], 
			const PetscMPIInt rank);

};

#endif
