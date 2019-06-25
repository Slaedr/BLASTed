/** \file
 * \brief Computes LHS and RHS for Poisson problem.
 */

#include <cassert>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "poisson3d_fd.hpp"
#include "utils/cmdoptions.hpp"
#include "poisson_setup.h"

#define PETSCOPTION_STR_LEN 30

#ifdef __cplusplus
extern "C" {
#endif

DiscreteLinearProblem setup_poisson_problem(const char *const confile)
{
	using namespace std; using blasted::petsc_throw;

	PetscMPIInt size, rank;
	PetscErrorCode ierr = 0;

	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&size);
	MPI_Comm_rank(comm,&rank);
	if(rank == 0)
		printf("Number of MPI ranks = %d.\n", size);

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
	if(rank == 0)
		printf("Max OMP threads = %d\n", nthreads);
#endif

	// Read control file
	
	PetscInt npdim[NDIM];
	PetscReal rmax[NDIM], rmin[NDIM];
	char temp[50], gridtype[50];
	FILE* conf = fopen(confile, "r");
	int fstatus = 1;
	fstatus = fscanf(conf, "%s", temp);
	fstatus = fscanf(conf, "%s", gridtype);
	fstatus = fscanf(conf, "%s", temp);
	if(!fstatus) {
		std::printf("! Error reading control file!\n");
		std::abort();
	}
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%d", &npdim[i]);
	fstatus = fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%lf", &rmin[i]);
	fstatus = fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fstatus = fscanf(conf, "%lf", &rmax[i]);
	fclose(conf);
	
	if(!fstatus) {
		std::printf("! Error reading control file!\n");
		std::abort();
	}

	if(rank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f ", rmin[i], rmax[i]);
		printf("\n");
	}

	// set up Petsc variables
	DM da;                        ///< Distributed array context for the cart grid
	PetscInt ndofpernode = 1;
	PetscInt stencil_width = 1;
	DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
	DMBoundaryType by = DM_BOUNDARY_GHOSTED;
	DMBoundaryType bz = DM_BOUNDARY_GHOSTED;
	DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

	// grid structure - a copy of the mesh is stored by all processes as the mesh structure is very small
	CartMesh m;
	ierr = m.createMeshAndDMDA(comm, npdim, ndofpernode, stencil_width, bx, by, bz, stencil_type, 
	                           &da, rank);
	petsc_throw(ierr);

	// generate grid
	if(!strcmp(gridtype, "chebyshev"))
		m.generateMesh_ChebyshevDistribution(rmin,rmax, rank);
	else
		m.generateMesh_UniformDistribution(rmin,rmax, rank);

	DiscreteLinearProblem lp;

	// create vectors and matrix according to the DMDA's structure

	ierr = DMCreateGlobalVector(da, &lp.uexact); petsc_throw(ierr);
	ierr = VecDuplicate(lp.uexact, &lp.b); petsc_throw(ierr);
	ierr = DMCreateMatrix(da, &lp.lhs); petsc_throw(ierr);

	// compute values of LHS, RHS and exact soln

	ierr = computeRHS(&m, da, rank, lp.b, lp.uexact); petsc_throw(ierr);
	ierr = computeLHS(&m, da, rank, lp.lhs); petsc_throw(ierr);

	// Assemble LHS

	ierr = MatAssemblyBegin(lp.lhs, MAT_FINAL_ASSEMBLY); petsc_throw(ierr);
	ierr = MatAssemblyEnd(lp.lhs, MAT_FINAL_ASSEMBLY); petsc_throw(ierr);

	if(rank == 0) {
		printf(" Poisson problem assembled.\n");
		printf("  Mesh size h : %f\n", m.gh());
		printf("        log h : %f\n", log10(m.gh()));
	}
	return lp;
}

#ifdef __cplusplus
}
#endif
