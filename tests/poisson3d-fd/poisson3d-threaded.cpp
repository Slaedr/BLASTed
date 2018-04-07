/** Checks correctness of the solution computed by threaded async preconditioners.
 */

#include <petscksp.h>

#include <sys/time.h>
#include <ctime>
#include <cfloat>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <blasted_petsc.h>

#include "poisson3d_fd.hpp"

#define PETSCOPTION_STR_LEN 30

PetscReal compute_error(const MPI_Comm comm, const CartMesh& m, const DM da,
		const Vec u, const Vec uexact) {
	PetscReal errnorm;
	Vec err;
	VecDuplicate(u, &err);
	VecCopy(u,err);
	VecAXPY(err, -1.0, uexact);
	errnorm = computeNorm(comm, &m, err, da);
	VecDestroy(&err);
	return errnorm;
}

int main(int argc, char* argv[])
{
	using namespace std;

	if(argc < 3) {
		printf("Please specify a control file and a Petsc options file.\n");
		return 0;
	}

	char help[] = "Solves 3D Poisson equation by finite differences.\
				   Arguments: (1) Control file (2) Petsc options file\n\n";
	char * confile = argv[1];
	char * optfile = argv[2];
	PetscMPIInt size, rank;
	PetscErrorCode ierr = 0;
	int nruns;

	ierr = PetscInitialize(&argc, &argv, optfile, help); CHKERRQ(ierr);
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
	fstatus = fscanf(conf, "%s", temp); 
	fstatus = fscanf(conf, "%d", &nruns);
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
		printf("Number of runs: %d\n", nruns);
	}
	//----------------------------------------------------------------------------------

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
	CHKERRQ(ierr);

	// generate grid
	if(!strcmp(gridtype, "chebyshev"))
		m.generateMesh_ChebyshevDistribution(rmin,rmax, rank);
	else
		m.generateMesh_UniformDistribution(rmin,rmax, rank);

	Vec u, uexact, b, err;
	Mat A;

	// create vectors and matrix according to the DMDA's structure
	
	ierr = DMCreateGlobalVector(da, &u); CHKERRQ(ierr);
	ierr = VecDuplicate(u, &b); CHKERRQ(ierr);
	VecDuplicate(u, &uexact);
	VecDuplicate(u, &err);
	VecSet(u, 0.0);
	ierr = DMCreateMatrix(da, &A); CHKERRQ(ierr);

	// compute values of LHS, RHS and exact soln
	
	ierr = computeRHS(&m, da, rank, b, uexact); CHKERRQ(ierr);
	ierr = computeLHS(&m, da, rank, A); CHKERRQ(ierr);

	// Assemble LHS

	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	KSP kspref; 

	// compute reference solution using a preconditioner from PETSc
	
	ierr = KSPCreate(comm, &kspref);
	KSPSetType(kspref, KSPRICHARDSON);
	KSPRichardsonSetScale(kspref, 1.0);
	KSPSetOptionsPrefix(kspref, "ref_");
	KSPSetFromOptions(kspref);
	
	ierr = KSPSetOperators(kspref, A, A); CHKERRQ(ierr);
	
	ierr = KSPSolve(kspref, b, u); CHKERRQ(ierr);

	PetscInt refkspiters;
	ierr = KSPGetIterationNumber(kspref, &refkspiters);
	PetscReal errnormref = compute_error(comm,m,da,u,uexact);

	if(rank==0) {
		printf("Ref run: error = %.16f\n", errnormref);
	}

	KSPDestroy(&kspref);

	// run the solve to be tested as many times as requested
	
	int avgkspiters = 0;
	PetscReal errnorm = 0;
	
	// test with 4 threads
#ifdef _OPENMP
	omp_set_num_threads(4);
#endif
	
	for(int irun = 0; irun < nruns; irun++)
	{
		if(rank == 0)
			printf("Run %d:\n", irun);
		KSP ksp;

		ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
		KSPSetType(ksp, KSPRICHARDSON);
		KSPRichardsonSetScale(ksp, 1.0);
	
		// Options MUST be set before setting shell routines!
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		// Operators MUST be set before extracting sub KSPs!
		ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
		
		// Create BLASTed data structure and setup the PC
		Blasted_data bctx = newBlastedDataContext();
		ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);
		
		ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		KSPGetIterationNumber(ksp, &kspiters);
		avgkspiters += kspiters;

		if(rank == 0) {
			//printf(" Number of KSP iterations = %d\n", kspiters);
			KSPGetResidualNorm(ksp, &rnorm);
			printf(" KSP residual norm = %f, num iters = %d.\n", rnorm, kspiters);
		}
		
		errnorm = compute_error(comm,m,da,u,uexact);
		if(rank == 0) {
			printf("Test run:\n");
			printf(" h and error: %f  %.16f\n", m.gh(), errnorm);
			printf(" log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));
		}
		
		// test
		if(std::fabs(errnorm-errnormref) > 10.0*DBL_EPSILON) {
			printf("Difference in error norm = %.16f.\n", std::fabs(errnorm-errnormref));
			return -1;
		}

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	}

	if(rank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters/nruns);

	VecDestroy(&u);
	VecDestroy(&uexact);
	VecDestroy(&b);
	VecDestroy(&err);
	MatDestroy(&A);
	DMDestroy(&da);
	PetscFinalize();

	return ierr;
}
		
// some unused snippets that might be useful at some point
	
	/*struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;*/

	/*gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	if(rank==0) {
		printf("Total times: Wall = %f, CPU = %f\n", finalwtime-initialwtime, finalctime-initialctime);
		printf("Time taken by FGPILU factorization: Wall = %f, CPU = %f\n", iluctrl.factorwalltime, iluctrl.factorcputime);
		printf("Time taken by FGPILU application: Wall = %f, CPU = %f\n", iluctrl.applywalltime, iluctrl.applycputime);
		double totalwall = iluctrl.factorwalltime + iluctrl.applywalltime;
		double totalcpu = iluctrl.factorcputime + iluctrl.applycputime;
		printf("Time taken by FGPILU total: Wall = %f, CPU = %f\n", totalwall, totalcpu);
		printf("Average number of iterations: %d\n", (int)(avgkspiters/nruns));
	}*/

/* For viewing the ILU factors computed by PETSc PCILU
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/ksp/pc/impls/factor/factor.h>
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>*/

	/*if(precch == 'i') {	
		// view factors
		PC_ILU* ilu = (PC_ILU*)pc->data;
		//PC_Factor* pcfact = (PC_Factor*)pc->data;
		//Mat fact = pcfact->fact;
		Mat fact = ((PC_Factor*)ilu)->fact;
		printf("ILU0 factored matrix:\n");

		Mat_SeqAIJ* fseq = (Mat_SeqAIJ*)fact->data;
		for(int i = 0; i < fact->rmap->n; i++) {
			printf("Row %d: ", i);
			for(int j = fseq->i[i]; j < fseq->i[i+1]; j++)
				printf("(%d: %f) ", fseq->j[j], fseq->a[j]);
			printf("\n");
		}
	}*/

