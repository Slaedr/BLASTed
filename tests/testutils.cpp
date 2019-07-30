/** \file
 * \brief Implementation of some testing utilities
 */

#undef NDEBUG
#include <cassert>
#include <stdexcept>
#include <string>
#include <cmath>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <petscksp.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

#include "utils/mpiutils.hpp"
#include "utils/cmdoptions.hpp"
#include "preconditioner_diagnostics.hpp"
#include "blasted_petsc.h"
#include "testutils.h"
#include "testutils.hpp"

#define PETSCOPTION_STR_LEN 30

namespace blasted {

SRMatrixStorage<const PetscScalar,const PetscInt> wrapLocalPetscMat(Mat A, const int bs)
{
	PetscInt firstrow, lastrow, localrows, localcols, globalrows, globalcols;
	int ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); petsc_throw(ierr);
	ierr = MatGetLocalSize(A, &localrows, &localcols); petsc_throw(ierr);
	ierr = MatGetSize(A, &globalrows, &globalcols); petsc_throw(ierr);
	assert(localrows == localcols);
	assert(globalrows == globalcols);

	// get access to local matrix entries
	if(bs == 1) {
		const Mat_SeqAIJ *const Adiag = (const Mat_SeqAIJ*)A->data;
		assert(Adiag != NULL);

		return SRMatrixStorage<const PetscScalar, const PetscInt>(Adiag->i, Adiag->j, Adiag->a,
		                                                          Adiag->diag, Adiag->i + 1,
		                                                          localrows, Adiag->i[localrows],
		                                                          Adiag->i[localrows]);
	}
	else {
		const Mat_SeqBAIJ *const Adiag = (const Mat_SeqBAIJ*)A->data;
		assert(Adiag != NULL);

		return SRMatrixStorage<const PetscScalar, const PetscInt>(Adiag->i, Adiag->j, Adiag->a,
		                                                          Adiag->diag, Adiag->i + 1,
		                                                          localrows, Adiag->i[localrows],
		                                                          Adiag->i[localrows]);
	}
}

extern "C" {

int compareSolverWithRef(const int refkspiters, const int avgkspiters,
                         Vec uref, Vec u)
{
	const int rank = get_mpi_rank(MPI_COMM_WORLD);
	if(rank == 0)
		printf("KSP Iters: Reference %d vs BLASTed %d.\n", refkspiters, avgkspiters);
	fflush(stdout);

	const std::string testtype = parsePetscCmd_string("-test_type", PETSCOPTION_STR_LEN);
	const double error_tol = parseOptionalPetscCmd_real("-error_tolerance", 2*DBL_EPSILON);
	//const double iters_tol = parseOptionalPetscCmd_real("-iters_tolerance", 1e-2);

	if(rank == 0)
		printf("  Test tolerance = %g.\n", error_tol);

	if(testtype == "compare_its" || testtype == "issame") {
		assert(fabs((double)refkspiters - avgkspiters)/refkspiters <= error_tol);
	}
	else if(testtype == "upper_bound_its") {
		assert(refkspiters > avgkspiters);
	}

	Vec diff;
	int ierr = VecDuplicate(u, &diff); CHKERRQ(ierr);
	ierr = VecSet(diff, 0); CHKERRQ(ierr);
	ierr = VecWAXPY(diff, -1.0, u, uref); CHKERRQ(ierr);
	PetscScalar diffnorm, refnorm;
	ierr = VecNorm(uref, NORM_2, &refnorm); CHKERRQ(ierr);
	ierr = VecNorm(diff, NORM_2, &diffnorm); CHKERRQ(ierr);
	ierr = VecDestroy(&diff); CHKERRQ(ierr);

	printf("Difference in solutions = %g.\n", diffnorm);
	printf("Relative difference = %g.\n", diffnorm/refnorm);
	fflush(stdout);
	if(testtype == "compare_error" || testtype == "issame")
		assert(diffnorm/refnorm <= error_tol);

	return 0;
}

static int pretendModifyMatrix(const Mat A)
{
	int firstrow, lastrow;
	int ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); CHKERRQ(ierr);

	// int ncols;
	// const int *cols;
	// const double *vals;
	// ierr = MatGetRow(A, firstrow, &ncols, &cols, &vals); CHKERRQ(ierr);
	// const double firstval = vals[0];
	// const int firstcol = cols[0];
	// ierr = MatRestoreRow(A, firstrow, &ncols, &cols, &vals); CHKERRQ(ierr);

	// double firstval = 0;
	// ierr = MatGetValues(A, 1, &firstrow, 1, &firstrow, &firstval); CHKERRQ(ierr);

	// ierr = MatSetValues(A, 1, &firstrow, 1, &firstrow, &firstval, INSERT_VALUES); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	return ierr;
}

int runComparisonVsPetsc_cpp(const DiscreteLinearProblem lp)
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

#ifdef _OPENMP
	const int nthreads = omp_get_max_threads();
#endif

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

	// Create BLASTed data structure and setup the PC
	Blasted_data_list bctx = newBlastedDataList();

	ierr = MatSetOption(lp.lhs, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE); CHKERRQ(ierr);

	KSP ksp;

	ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
	ierr = KSPSetType(ksp, KSPRICHARDSON); CHKERRQ(ierr);
	ierr = KSPRichardsonSetScale(ksp, 1.0); CHKERRQ(ierr);

	// Options MUST be set before setting shell routines!
	ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
	// Operators MUST be set before extracting sub KSPs!
	ierr = KSPSetOperators(ksp, lp.lhs, lp.lhs); CHKERRQ(ierr);

	ierr = setup_blasted_stack(ksp, &bctx); CHKERRQ(ierr);

	for(int irun = 0; irun < nruns; irun++)
	{
		if(rank == 0)
			printf("Run %d:\n", irun);

		Vec urun;
		ierr = VecDuplicate(lp.uexact, &urun); CHKERRQ(ierr);

		ierr = pretendModifyMatrix(lp.lhs); CHKERRQ(ierr);

		ierr = KSPSolve(ksp, lp.b, urun); CHKERRQ(ierr);

		// post-process
		int kspiters; PetscReal rnorm;
		ierr = KSPGetIterationNumber(ksp, &kspiters); CHKERRQ(ierr);
		avgkspiters += kspiters;

		KSPConvergedReason ksp_reason;
		ierr = KSPGetConvergedReason(ksp, &ksp_reason); CHKERRQ(ierr);
		printf("  KSP converged reason = %d.\n", ksp_reason); fflush(stdout);
		assert(ksp_reason > 0);

		if(rank == 0) {
			ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
			printf(" KSP residual norm = %f\n", rnorm);
		}

		PetscReal errnormrun = 0;
		ierr = compute_difference_norm(urun, lp.uexact, &errnormrun); CHKERRQ(ierr);

		ierr = VecAXPY(u, 1.0, urun); CHKERRQ(ierr);

		if(rank == 0) {
			printf("Test run:\n");
			printf(" error: %.16f\n", errnormrun);
			printf(" log error: %f\n", log10(errnormrun));
		}

		ierr = VecDestroy(&urun); CHKERRQ(ierr);

		// rudimentary test for time-totaller
		computeTotalTimes(&bctx);
		assert(bctx.factorwalltime > DBL_EPSILON);
		assert(bctx.applywalltime > DBL_EPSILON);
		// looks like the problem is too small for the unix clock() to record it
		assert(bctx.factorcputime >= 0);
		assert(bctx.applycputime >= 0);
	}

	const PrecInfoList *const pilist = static_cast<const PrecInfoList*>(bctx.ctxlist->infolist);

	bool testprecinfo;
	try {
		testprecinfo = parsePetscCmd_bool("-blasted_compute_preconditioner_info");
	} catch (InputNotGivenError& e) {
		testprecinfo = false;
	}

	if(testprecinfo) {
		assert(pilist);
		printf(" Size if infolist is %ld, num runs is %d.\n", pilist->infolist.size(), nruns);
		fflush(stdout);
		// It turns out one extra call to the PC's compute function is made during precprocessing
		assert(static_cast<int>(pilist->infolist.size()) == nruns+1);

		printf("  Preconditioner build info:\n");
		printf("  ILU rem, initial ILU rem, upper avg ddom, upper min ddom, lower avg ddom, lower min ddom\n");
		for(int i = 0; i < nruns+1; i++) {
			for(int j = 0; j < 6; j++)
				printf("  %f ", pilist->infolist[i].f_info[j]);
			printf("\n");
			fflush(stdout);

			// Sanity checks
			for(int j = 0; j < 6; j++)
				assert(std::isfinite(pilist->infolist[i].f_info[j]));
			assert(pilist->infolist[i].prec_rem_initial_norm() > 0);
			assert(pilist->infolist[i].prec_remainder_norm() < 100.0);
			if(nthreads == 1)
				assert(pilist->infolist[i].prec_remainder_norm() < 1e-14);
			assert(pilist->infolist[i].upper_avg_diag_dom() < 1);
			assert(pilist->infolist[i].upper_min_diag_dom() <= 1);
			assert(pilist->infolist[i].lower_avg_diag_dom() < 1);
			assert(pilist->infolist[i].lower_min_diag_dom() <= 1);
		}
	}
	else {
		assert(pilist == nullptr);
	}

	ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

	destroyBlastedDataList(&bctx);

	avgkspiters = avgkspiters/(double)nruns;
	ierr = VecScale(u, 1.0/nruns); CHKERRQ(ierr);

	ierr = compareSolverWithRef(refkspiters, avgkspiters, uref, u); CHKERRQ(ierr);

	ierr = VecDestroy(&u); CHKERRQ(ierr);
	ierr = VecDestroy(&uref); CHKERRQ(ierr);

	return ierr;
}

}
}

// some unused snippets that might be useful at some point

/* Wall clock time by Unix functions

struct timeval time1, time2;
gettimeofday(&time1, NULL);
double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

some code here..

gettimeofday(&time2, NULL);
double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
*/

/* For viewing the ILU factors computed by PETSc PCILU

#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/ksp/pc/impls/factor/factor.h>
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>

if(precch == 'i') {	
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
}
*/

