/** \file
 * \brief Computes LHS and RHS for Poisson problem.
 */

#include <cassert>
#include <stdexcept>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "poisson3d_fd.hpp"
#include "poisson_utils.hpp"
#include "utils/cmdoptions.hpp"
#include "poisson_setup.h"

#define PETSCOPTION_STR_LEN 30

namespace test {

MeshSpec readInputFile(const MPI_Comm comm, const char *confile)
{
	using blasted::file_input_throw;
	int rank;
	MPI_Comm_rank(comm,&rank);

	MeshSpec spec;

	char temp[50];
	FILE* conf = fopen(confile, "r");
	int fstatus = 1;
	fstatus = fscanf(conf, "%s", temp); file_input_throw(fstatus);
	fstatus = fscanf(conf, "%s", spec.gridtype); file_input_throw(fstatus);

	fstatus = fscanf(conf, "%s", temp); file_input_throw(fstatus);
	for(int i = 0; i < NDIM; i++) {
		fstatus = fscanf(conf, "%d", &spec.npdim[i]);
		file_input_throw(fstatus);
	}

	fstatus = fscanf(conf, "%s", temp); file_input_throw(fstatus);
	for(int i = 0; i < NDIM; i++) {
		fstatus = fscanf(conf, "%lf", &spec.rmin[i]);
		file_input_throw(fstatus);
	}

	fstatus = fscanf(conf, "%s", temp); file_input_throw(fstatus);
	for(int i = 0; i < NDIM; i++) {
		fstatus = fscanf(conf, "%lf", &spec.rmax[i]);
		file_input_throw(fstatus);
	}
	fclose(conf);

	if(rank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f ", spec.rmin[i], spec.rmax[i]);
		printf("\n");
	}

	return spec;
}

CartMesh createMesh(MPI_Comm comm, const MeshSpec ms)
{
	int ierr, rank;
	MPI_Comm_rank(comm,&rank);

	// set up Petsc variables
	PetscInt ndofpernode = 1;
	PetscInt stencil_width = 1;
	DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
	DMBoundaryType by = DM_BOUNDARY_GHOSTED;
	DMBoundaryType bz = DM_BOUNDARY_GHOSTED;
	DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

	// grid structure - a copy of the mesh is stored by all processes as the mesh structure is very small
	CartMesh m;
	ierr = m.createMeshAndDMDA(comm, ms.npdim, ndofpernode, stencil_width, bx, by, bz, stencil_type);
	blasted::petsc_throw(ierr);

	// generate grid
	if(!strcmp(ms.gridtype, "chebyshev"))
		m.generateMesh_ChebyshevDistribution(ms.rmin, ms.rmax, rank);
	else
		m.generateMesh_UniformDistribution(ms.rmin, ms.rmax, rank);

	return m;
}

}

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
	const test::MeshSpec ms = test::readInputFile(comm, confile);
	const CartMesh m = test::createMesh(comm, ms);
	const DM da = m.getDA();

	DiscreteLinearProblem lp;

	// create vectors and matrix according to the DMDA's structure

	ierr = DMCreateGlobalVector(da, &lp.uexact); petsc_throw(ierr);
	ierr = VecDuplicate(lp.uexact, &lp.b); petsc_throw(ierr);
	ierr = DMCreateMatrix(da, &lp.lhs); petsc_throw(ierr);

	// compute values of LHS, RHS and exact soln

	ierr = computeRHS(&m, rank, lp.b, lp.uexact); petsc_throw(ierr);
	ierr = computeLHS(&m, rank, lp.lhs); petsc_throw(ierr);

	// Assemble LHS

	ierr = MatAssemblyBegin(lp.lhs, MAT_FINAL_ASSEMBLY); petsc_throw(ierr);
	ierr = MatAssemblyEnd(lp.lhs, MAT_FINAL_ASSEMBLY); petsc_throw(ierr);

	// PetscBool diagmissing = PETSC_FALSE;
	// PetscInt badrow = -1;
	// ierr = MatMissingDiagonal(lp.lhs, &diagmissing, &badrow); CHKERRQ(ierr);
	// if(diagmissing == PETSC_TRUE) {
	// 	throw std::runtime_error("! Zero diagonal in (block-)row " + std::to_string(badrow));
	// }

	if(rank == 0) {
		printf(" Poisson problem assembled.\n");
		printf("  Mesh size h : %f\n", m.gh());
		printf("        log h : %f\n", log10(m.gh()));
	}
	return lp;
}

DiscreteLinearProblem generateDiscreteProblem(const int argc, char *argv[], const int argstart)
{
	DiscreteLinearProblem dlp;
	if(!strcmp(argv[argstart],"poisson"))
	{
		if(argc < 3) {
			printf(" ! Please provide a Poisson control file!\n");
			exit(-1);
		}

		dlp = setup_poisson_problem(argv[argstart+1]);
	}
	else {
		if(argc < 5) {
			printf(" ! Please provide filenames for LHS, RHS vector and exact solution (in order).\n");
			exit(-1);
		}

		int ierr = readLinearSystemFromFiles(argv[argstart+1], argv[argstart+2], argv[argstart+3], &dlp);
		if(ierr)
			throw std::runtime_error("Could not read linear system from files!");
	}

	return dlp;
}

#ifdef __cplusplus
}
#endif
