/** \file cartmesh.cpp
 * \brief Implementation of single-block but distributed, non-uniform Cartesian grid using PETSc
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

#include "cartmesh.hpp"

void CartMesh::computeMeshSize()
{
	// estimate h
	h = 0.0;
	PetscReal hd[NDIM];
	for(int k = 0; k < npoind[2]-1; k++)
	{
		hd[2] = coords[2][k+1]-coords[2][k];
		for(int j = 0; j < npoind[1]-1; j++)
		{
			hd[1] = coords[1][j+1]-coords[1][j];
			for(int i = 0; i < npoind[0]-1; i++)
			{
				hd[0] = coords[0][i+1]-coords[0][i];
				PetscReal diam = 0;
				for(int idim = 0; idim < NDIM; idim++)
					diam += hd[idim]*hd[idim];
				diam = std::sqrt(diam);
				if(diam > h)
					h = diam;
			}
		}
	}
}

CartMesh::CartMesh()
	: coords{NULL}, da{NULL}
{ }

CartMesh::CartMesh(const PetscInt npdim[NDIM], const PetscInt num_partitions)
	: coords{NULL}, da{NULL}
{
	std::printf("CartMesh: Number of points in each direction: ");
	for(int i = 0; i < NDIM; i++) {
		npoind[i] = npdim[i];
		std::printf("%d ", npoind[i]);
	}
	std::printf("\n");
	
	npointotal = 1;
	for(int i = 0; i < NDIM; i++)
		npointotal *= npoind[i];

	PetscInt nbpoints = npoind[0]*npoind[1]*2 + (npoind[2]-2)*npoind[0]*2 + 
		(npoind[1]-2)*(npoind[2]-2)*2;
	ninpoin = npointotal-nbpoints;

	std::printf("CartMesh: Total points = %d, interior points = %d\n", npointotal, ninpoin);
}

PetscErrorCode CartMesh::createMeshAndDMDA(const MPI_Comm comm, const PetscInt npdim[NDIM], 
                                           PetscInt ndofpernode, PetscInt stencil_width,
                                           DMBoundaryType bx, DMBoundaryType by, DMBoundaryType bz,
                                           DMDAStencilType stencil_type)
{
	PetscErrorCode ierr = 0;
	int rank;
	MPI_Comm_rank(comm, &rank);

	for(int i = 0; i < NDIM; i++) {
		npoind[i] = npdim[i];
	}

	if(rank == 0) {
		std::printf("CartMesh: Number of points in each direction: ");
		for(int i = 0; i < NDIM; i++) {
			std::printf("%d ", npoind[i]);
		}
		std::printf("\n");
	}

	npointotal = 1;
	for(int i = 0; i < NDIM; i++)
		npointotal *= npoind[i];

	PetscInt nbpoints = npoind[0]*npoind[1]*2 + (npoind[2]-2)*npoind[0]*2 + 
		(npoind[1]-2)*(npoind[2]-2)*2;
	ninpoin = npointotal-nbpoints;

	if(rank == 0)
		std::printf("CartMesh: Setting up DMDA\n");

	ierr = DMDACreate3d(comm, bx, by, bz, stencil_type,
	                    npoind[0]-2, npoind[1]-2, npoind[2]-2,
	                    PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, ndofpernode, stencil_width,
	                    NULL, NULL, NULL, &da);
	CHKERRQ(ierr);
	ierr = DMSetUp(da); CHKERRQ(ierr);

	PetscInt M,N,P;
	ierr = DMDAGetInfo(da, NULL, &M, &N, &P, &nprocs[0], &nprocs[1], &nprocs[2],
	                   NULL, NULL, NULL, NULL, NULL, NULL);
	CHKERRQ(ierr);

	ntprocs = nprocs[0]*nprocs[1]*nprocs[2];

	if(rank == 0) {
		std::printf("CartMesh: Number of points in each direction: %d,%d,%d.\n",
		            M,N,P);
		std::printf("CartMesh: Number of procs in each direction: %d,%d,%d.\n",
		            nprocs[0], nprocs[1], nprocs[2]);
		std::printf("CartMesh: Total points = %d, interior points = %d, partitions = %d\n",
		            npointotal, ninpoin, ntprocs);
	}

	// have each process store coords; hardly costs anything
	coords = (PetscReal**)std::malloc(NDIM*sizeof(PetscReal*));
	for(int i = 0; i < NDIM; i++)
		coords[i] = (PetscReal*)std::malloc(npoind[i]*sizeof(PetscReal));

	return ierr;
}

CartMesh::~CartMesh()
{
	int ierr = DMDestroy(&da);
	if(ierr)
		std::printf("Could not destroy DM!\n");
	for(int i = 0; i < NDIM; i++)
		std::free(coords[i]);
	std::free(coords);
}

void CartMesh::generateMesh_ChebyshevDistribution(const PetscReal rmin[NDIM],
                                                  const PetscReal rmax[NDIM],
                                                  const PetscMPIInt rank)
{
	if(rank == 0)
		std::printf("CartMesh: generateMesh_cheb: Generating grid\n");
	for(int idim = 0; idim < NDIM; idim++)
	{
		PetscReal theta = PI/(npoind[idim]-1);
		for(int i = 0; i < npoind[idim]; i++) {
			coords[idim][i] = (rmax[idim]+rmin[idim])*0.5 + 
				(rmax[idim]-rmin[idim])*0.5*std::cos(PI-i*theta);
		}
	}

	// estimate h
	computeMeshSize();
	if(rank == 0)
		std::printf("CartMesh: generateMesh_Cheb: h = %f\n", h);
}

/// Generates grid with uniform spacing
void CartMesh::generateMesh_UniformDistribution(const PetscReal rmin[NDIM],
                                                const PetscReal rmax[NDIM],
                                                const PetscMPIInt rank)
{
	if(rank == 0)
		std::printf("CartMesh: generateMesh_Uniform: Generating grid\n");
	for(int idim = 0; idim < NDIM; idim++)
	{
		for(int i = 0; i < npoind[idim]; i++) {
			coords[idim][i] = rmin[idim] + (rmax[idim]-rmin[idim])*i/(npoind[idim]-1);
		}
	}

	computeMeshSize();
	if(rank == 0)
		std::printf("CartMesh: generateMesh_Uniform: h = %f\n", h);
}

