/** \file
 * \brief Implementation of some testing utilities
 */

#undef NDEBUG
#include <assert.h>
#include <float.h>
#include <math.h>

#include <petscksp.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

#include "blasted_petsc.h"
#include "testutils.h"

#define PETSCOPTION_STR_LEN 30

#define TASSERT(x) if(!(x)) return -1

int destroyDiscreteLinearProblem(DiscreteLinearProblem *const dlp)
{
	int ierr = MatDestroy(&(dlp->lhs)); CHKERRQ(ierr);
	ierr = VecDestroy(&(dlp->b)); CHKERRQ(ierr);
	ierr = VecDestroy(&(dlp->uexact)); CHKERRQ(ierr);
	return ierr;
}

int readLinearSystemFromFiles(const char *const matfile, const char *const bfile, const char *const xfile,
                              DiscreteLinearProblem *const lp, const bool testmatmult)
{
	PetscErrorCode ierr = 0;
	PetscMPIInt rank;
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);

	PetscViewer matreader;
	ierr = PetscViewerBinaryOpen(comm, matfile, FILE_MODE_READ, &matreader); CHKERRQ(ierr);
	PetscViewer bvecreader;
	ierr = PetscViewerBinaryOpen(comm, bfile, FILE_MODE_READ, &bvecreader); CHKERRQ(ierr);
	PetscViewer xvecreader;
	if(xfile) {
		ierr = PetscViewerBinaryOpen(comm, xfile, FILE_MODE_READ, &xvecreader); CHKERRQ(ierr);
	}

	ierr = MatCreate(comm,&(lp->lhs)); CHKERRQ(ierr);
	ierr = MatSetFromOptions(lp->lhs); CHKERRQ(ierr);
	ierr = MatLoad(lp->lhs, matreader); CHKERRQ(ierr);

	ierr = VecCreate(comm,&(lp->b)); CHKERRQ(ierr);
	ierr = VecLoad(lp->b, bvecreader); CHKERRQ(ierr);
	if(xfile) {
		ierr = VecCreate(comm,&(lp->uexact)); CHKERRQ(ierr);
		ierr = VecLoad(lp->uexact, xvecreader); CHKERRQ(ierr);
	}
	else
		lp->uexact = NULL;

	ierr = PetscViewerDestroy(&matreader); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&bvecreader); CHKERRQ(ierr);
	if(xfile) {
		ierr = PetscViewerDestroy(&xvecreader); CHKERRQ(ierr);
	}

	// ensure diagonal entry locations have been computed; this is necessary for BAIJ matrices
	// as a bonus, check for singular diagonals
	PetscBool diagmissing = PETSC_FALSE;
	PetscInt badrow = -1;
	ierr = MatMissingDiagonal(lp->lhs, &diagmissing, &badrow); CHKERRQ(ierr);
	if(diagmissing == PETSC_TRUE) {
		SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "! Zero diagonal in (block-)row %d!", badrow);
	}

	// Matrix multiply check

	if(testmatmult && xfile) {
		const PetscReal error_tol_matmult = 5e2;

		Vec test;
		ierr = MatCreateVecs(lp->lhs, NULL, &test); CHKERRQ(ierr);
		PetscInt vs;
		ierr = VecGetLocalSize(test, &vs); CHKERRQ(ierr);
		printf(" Rank %d: Local test size = %d.\n", rank, vs);
		ierr = VecGetLocalSize(lp->uexact, &vs); CHKERRQ(ierr);
		printf(" Rank %d: Local u_exact size = %d.\n", rank, vs);
		ierr = MatMult(lp->lhs, lp->uexact, test); CHKERRQ(ierr);
		ierr = VecAXPY(test, -1.0, lp->b); CHKERRQ(ierr);
		PetscReal multerrnorm = 0;
		ierr = VecNorm(test, NORM_2, &multerrnorm); CHKERRQ(ierr);
		printf(" Mult error = %16.16e\n", multerrnorm);
		TASSERT(multerrnorm < error_tol_matmult*DBL_EPSILON);
		ierr = VecDestroy(&test); CHKERRQ(ierr);
	}

	return ierr;
}

int compute_difference_norm(const Vec u, const Vec uexact, PetscReal *const errnorm)
{
	Vec err;
	int ierr = VecDuplicate(u, &err); CHKERRQ(ierr);
	VecCopy(u,err);
	VecAXPY(err, -1.0, uexact);

	*errnorm = 0;
	PetscInt numpoints = 0;
	ierr = VecGetSize(u, &numpoints); CHKERRQ(ierr);
	ierr = VecNorm(err, NORM_2, errnorm); CHKERRQ(ierr);
	*errnorm = *errnorm / sqrt(numpoints);
	ierr = VecDestroy(&err); CHKERRQ(ierr);
	return ierr;
}

int runComparisonVsPetsc(const DiscreteLinearProblem lp)
{
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	char testtype[PETSCOPTION_STR_LEN];
	PetscBool set = PETSC_FALSE;
	int ierr = PetscOptionsGetString(NULL,NULL,"-test_type",testtype, PETSCOPTION_STR_LEN, &set);
	CHKERRQ(ierr);
	if(!set) {
		printf("Test type not set; testing issame.\n");
		strcpy(testtype,"issame");
	}

	set = PETSC_FALSE;
	PetscInt cmdnumruns;
	ierr = PetscOptionsGetInt(NULL,NULL,"-num_runs",&cmdnumruns,&set); CHKERRQ(ierr);
	const int nruns = set ? cmdnumruns : 1;

	//----------------------------------------------------------------------------------

	// compute reference solution using a preconditioner from PETSc

	Vec uref;
	ierr = VecDuplicate(lp.uexact, &uref); CHKERRQ(ierr);

	KSP kspref;
	ierr = KSPCreate(PETSC_COMM_WORLD, &kspref);
	KSPSetType(kspref, KSPRICHARDSON);
	KSPRichardsonSetScale(kspref, 1.0);
	KSPSetOptionsPrefix(kspref, "ref_");
	KSPSetFromOptions(kspref);

	ierr = KSPSetOperators(kspref, lp.lhs, lp.lhs); CHKERRQ(ierr);

	ierr = KSPSolve(kspref, lp.b, uref); CHKERRQ(ierr);

	KSPConvergedReason ref_ksp_reason;
	ierr = KSPGetConvergedReason(kspref, &ref_ksp_reason); CHKERRQ(ierr);
	assert(ref_ksp_reason > 0);

	PetscInt refkspiters;
	ierr = KSPGetIterationNumber(kspref, &refkspiters);
	PetscReal errnormref = 0;
	ierr = compute_difference_norm(uref,lp.uexact,&errnormref); CHKERRQ(ierr);

	if(rank==0) {
		printf("Ref run: error = %.16f\n", errnormref);
	}

	ierr = KSPDestroy(&kspref); CHKERRQ(ierr);

	// run the solve to be tested as many times as requested

	int avgkspiters = 0;

	Vec u;
	ierr = VecDuplicate(lp.uexact, &u); CHKERRQ(ierr);
	ierr = VecSet(u, 0); CHKERRQ(ierr);

	for(int irun = 0; irun < nruns; irun++)
	{
		if(rank == 0)
			printf("Run %d:\n", irun);

		Vec urun;
		ierr = VecDuplicate(lp.uexact, &urun); CHKERRQ(ierr);

		KSP ksp;

		ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
		ierr = KSPSetType(ksp, KSPRICHARDSON); CHKERRQ(ierr);
		ierr = KSPRichardsonSetScale(ksp, 1.0); CHKERRQ(ierr);

		// Options MUST be set before setting shell routines!
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		// Operators MUST be set before extracting sub KSPs!
		ierr = KSPSetOperators(ksp, lp.lhs, lp.lhs); CHKERRQ(ierr);

		// Create BLASTed data structure and setup the PC
		Blasted_data_list bctx = newBlastedDataList();
		ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);

		ierr = KSPSolve(ksp, lp.b, urun); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		ierr = KSPGetIterationNumber(ksp, &kspiters); CHKERRQ(ierr);
		avgkspiters += kspiters;

		KSPConvergedReason ksp_reason;
		ierr = KSPGetConvergedReason(ksp, &ksp_reason); CHKERRQ(ierr);
		if(rank == 0)
			printf("  KSP converged reason = %d.\n", ksp_reason);
		assert(ksp_reason > 0);

		ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
		if(rank == 0) {
			printf(" KSP residual norm = %f\n", rnorm);
		}

		//errnorm += compute_error(comm,m,da,u,lp.uexact);
		PetscReal errnormrun = 0;
		ierr = compute_difference_norm(u, lp.uexact, &errnormrun); CHKERRQ(ierr);

		ierr = VecAXPY(u, 1.0, urun); CHKERRQ(ierr);

		if(rank == 0) {
			printf("Test run:\n");
			printf(" error: %.16f\n", errnormrun);
			printf(" log error: %f\n", log10(errnormrun));
		}

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
		ierr = VecDestroy(&urun); CHKERRQ(ierr);

		// rudimentary test for time-totaller
		computeTotalTimes(&bctx);
		assert(bctx.factorwalltime > DBL_EPSILON);
		assert(bctx.applywalltime > DBL_EPSILON);
		// looks like the problem is too small for the unix clock() to record it
		assert(bctx.factorcputime >= 0);
		assert(bctx.applycputime >= 0);

		destroyBlastedDataList(&bctx);
	}

	avgkspiters = avgkspiters/(double)nruns;
	ierr = VecScale(u, 1.0/nruns); CHKERRQ(ierr);

	ierr = compareSolverWithRef(refkspiters, avgkspiters, uref, u); CHKERRQ(ierr);

	ierr = VecDestroy(&u); CHKERRQ(ierr);
	ierr = VecDestroy(&uref); CHKERRQ(ierr);

	return ierr;
}

int runPetsc(const DiscreteLinearProblem lp, const int buildsweeps, const int applysweeps,
             const int nruns, int *const iters, double *const rel_dev)
{
	int rank = 0;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	double avgkspiters = 0;

	set_blasted_sweeps(buildsweeps, applysweeps);

	Vec u;
	int ierr = VecDuplicate(lp.uexact, &u); CHKERRQ(ierr);
	ierr = VecSet(u, 0); CHKERRQ(ierr);

	//std::vector<int> runiters(nruns);
	int runiters[nruns];

	bool someRunNotConverged = false;
	int irun = 0;
	for( ; irun < nruns; irun++)
	{
		if(rank == 0)
			printf(" Run %d:\n", irun);

		Vec urun;
		ierr = VecDuplicate(lp.uexact, &urun); CHKERRQ(ierr);

		KSP ksp;

		ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
		ierr = KSPSetType(ksp, KSPRICHARDSON); CHKERRQ(ierr);
		ierr = KSPRichardsonSetScale(ksp, 1.0); CHKERRQ(ierr);

		// Options MUST be set before setting shell routines!
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

		// Operators MUST be set before extracting sub KSPs!
		ierr = KSPSetOperators(ksp, lp.lhs, lp.lhs); CHKERRQ(ierr);

		// Create BLASTed data structure and setup the PC
		Blasted_data_list bctx = newBlastedDataList();
		ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);

		ierr = KSPSolve(ksp, lp.b, urun); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		ierr = KSPGetIterationNumber(ksp, &kspiters); CHKERRQ(ierr);
		avgkspiters += kspiters;
		runiters[irun] = kspiters;

		KSPConvergedReason ksp_reason;
		ierr = KSPGetConvergedReason(ksp, &ksp_reason); CHKERRQ(ierr);
		if(rank == 0)
			printf("  KSP converged reason = %d.\n", ksp_reason);
		if(ksp_reason == KSP_DIVERGED_ITS) {
			if(rank == 0)
				printf(" KSP did not converge!\n");
			irun++;
			someRunNotConverged = true;
			break;
		}
		else if(ksp_reason <= 0) {
			if(rank == 0)
				printf(" KSP did not converge!\n");
			avgkspiters = -1e9;
			irun++;
			someRunNotConverged = true;
			break;
		}

		if(rank == 0) {
			ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
			printf("  KSP residual norm = %f\n", rnorm);
		}

		//errnorm += compute_error(comm,m,da,u,lp.uexact);
		PetscReal errnormrun = 0;
		ierr = compute_difference_norm(u, lp.uexact, &errnormrun); CHKERRQ(ierr);

		ierr = VecAXPY(u, 1.0, urun); CHKERRQ(ierr);

		if(rank == 0) {
			printf(" Test run:\n");
			printf("  error: %.16f\n", errnormrun);
			printf("  log error: %f\n", log10(errnormrun));
		}

		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
		ierr = VecDestroy(&urun); CHKERRQ(ierr);

		// rudimentary test for time-totaller
		computeTotalTimes(&bctx);
		assert(bctx.factorwalltime > DBL_EPSILON);
		assert(bctx.applywalltime > DBL_EPSILON);
		// looks like the problem is too small for the unix clock() to record it
		assert(bctx.factorcputime >= 0);
		assert(bctx.applycputime >= 0);

		destroyBlastedDataList(&bctx);
	}

	avgkspiters = avgkspiters/(double)irun;

	if(rank == 0)
		printf(" Average iteration count = %g\n", avgkspiters);
	if(someRunNotConverged) {
		if(rank == 0)
			printf("IMPORTANT: One of the runs did not converge for the setting (%d,%d)\n",
				   buildsweeps, applysweeps);
	}
	else {
		assert(irun == nruns);
	}

	*iters = round(avgkspiters);

	double reldev = 0;
	for(int i = 0; i < irun; i++)
		reldev += (runiters[i]-avgkspiters)*(runiters[i]-avgkspiters);
	reldev = sqrt(reldev/(double)irun)/avgkspiters;
	*rel_dev = reldev;

	ierr = VecDestroy(&u); CHKERRQ(ierr);

	return ierr;
}

