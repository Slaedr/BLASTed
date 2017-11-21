#include <petscksp.h>

#include <sys/time.h>
#include <time.h>

#include "../../src/blasted_petsc.h"

#include "poisson3d_fd.hpp"

/* For viewing the ILU factors computed by PETSc PCILU
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/ksp/pc/impls/factor/factor.h>
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>*/

int main(int argc, char* argv[])
{
	using namespace std;

	if(argc < 3) {
		printf("Please specify a control file and a Petsc options file.\n");
		return 0;
	}

	char help[] = "Solves 3D Poisson equation by finite differences. Arguments: (1) Control file (2) Petsc options file\n\n";
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
	fscanf(conf, "%s", temp);
	fscanf(conf, "%s", gridtype);
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%d", &npdim[i]);
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%lf", &rmin[i]);
	fscanf(conf, "%s", temp);
	for(int i = 0; i < NDIM; i++)
		fscanf(conf, "%lf", &rmax[i]);
	fscanf(conf, "%s", temp); fscanf(conf, "%d", &nruns);
	fclose(conf);

	if(rank == 0) {
		printf("Domain boundaries in each dimension:\n");
		for(int i = 0; i < NDIM; i++)
			printf("%f %f ", rmin[i], rmax[i]);
		printf("\n");
		printf("Number of runs: %d\n", nruns);
	}
	//----------------------------------------------------------------------------------

	// set up Petsc variables
	DM da;					///< Distributed array context for the cart grid
	PetscInt ndofpernode = 1;
	PetscInt stencil_width = 1;
	DMBoundaryType bx = DM_BOUNDARY_GHOSTED;
	DMBoundaryType by = DM_BOUNDARY_GHOSTED;
	DMBoundaryType bz = DM_BOUNDARY_GHOSTED;
	DMDAStencilType stencil_type = DMDA_STENCIL_STAR;

	// generate mesh - a copy of the mesh is stored by all processes as the mesh structure is very small
	CartMesh m;
	ierr = m.createMeshAndDMDA(comm, npdim, ndofpernode, stencil_width, bx, by, bz, stencil_type, &da, rank);
	CHKERRQ(ierr);
	
	if(!strcmp(gridtype, "chebyshev"))
		m.generateMesh_ChebyshevDistribution(rmin,rmax, rank);
	else
		m.generateMesh_UniformDistribution(rmin,rmax, rank);

	Vec u, uexact, b, err;
	Mat A;
	KSP ksp; PC pc;

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
	
	/*Mat B;
	MatConvert(A, MATSEQDENSE, MAT_INITIAL_MATRIX, &B);
	MatView(B, PETSC_VIEWER_STDOUT_WORLD);
	MatDestroy(&B);*/
	
	/*struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;*/
	int avgkspiters = 0;

	// set up solver
	for(int irun = 0; irun < nruns; irun++)
	{
		printf("Run %d:\n", irun);
		ierr = KSPCreate(comm, &ksp);
		KSPSetType(ksp, KSPRICHARDSON);
		KSPRichardsonSetScale(ksp, 1.0);
		KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 100);
		KSPGetPC(ksp, &pc);

		/*PCSetType(pc, PCSOR);
		PCSORSetOmega(pc,1.0);
		PCSORSetIterations(pc, 1, 1);
		ierr = PCSORSetSymmetric(pc, SOR_SYMMETRIC_SWEEP); CHKERRQ(ierr);

		PCSetType(pc, PCILU);
		PCFactorSetLevels(pc, 0);
		PCFactorSetMatOrderingType(pc, MATORDERINGNATURAL);
		
		PCSetType(pc, PCNONE);*/

		PCShellSetSetUp(pc, &compute_preconditioner_blasted);
		PCShellSetApply(pc, &apply_local_blasted);
		PCShellSetDestroy(pc, &cleanup_blasted);
		
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
		
		ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
		
		ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);
		ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

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

		// post-process
		if(rank == 0) {
			int kspiters; PetscReal rnorm;
			KSPGetIterationNumber(ksp, &kspiters);
			printf(" Number of KSP iterations = %d\n", kspiters);
			avgkspiters += kspiters;
			KSPGetResidualNorm(ksp, &rnorm);
			printf(" KSP residual norm = %f\n", rnorm);
		}
		
		PetscReal errnorm;
		VecCopy(u,err);
		VecAXPY(err, -1.0, uexact);
		errnorm = computeNorm(comm, &m, err, da);
		if(rank == 0) {
			printf(" h and error: %f  %f\n", m.gh(), errnorm);
			printf(" log h and log error: %f  %f\n", log10(m.gh()), log10(errnorm));
		}

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	}

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

	VecDestroy(&u);
	VecDestroy(&uexact);
	VecDestroy(&b);
	VecDestroy(&err);
	MatDestroy(&A);
	DMDestroy(&da);
	PetscFinalize();
	return ierr;
}
