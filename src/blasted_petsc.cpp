/** \file blasted_petsc.cpp
 * \brief C wrapper for using BLASTed functionality with PETSc
 * \author Aditya Kashi
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#include "blockmatrices.hpp"

#include "blasted_petsc.h"

using namespace blasted;

typedef MatrixView<PetscReal, PetscInt> BlastedPetscMat;

#define PETSCOPTION_STR_LEN 10

/// Sets options from PETSc options
static PetscErrorCode setupDataFromOptions(PC pc)
{
	PetscErrorCode ierr=0;
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);

	Prec_type ptype;

	PetscBool set = PETSC_FALSE;
	PetscOptionsHasName(NULL, NULL, "-blasted_pc_type", &set);
	if(set == PETSC_FALSE) {
		printf("BLASTed: Preconditioner type not set! Setting to Jacobi.\n");
		ptype = JACOBI;
	}
	else {
		char precstr[PETSCOPTION_STR_LEN];
		PetscBool flag = PETSC_FALSE;
		PetscOptionsGetString(NULL, NULL, "-blasted_pc_type", 
				precstr, PETSCOPTION_STR_LEN, &flag);
		if(flag == PETSC_FALSE) {
			printf("BLASTed: Preconditioner type not set!\n");
			abort();
		}

		std::string precstr2 = precstr;
		if(precstr2 == "jacobi")
			ptype = JACOBI;
		else if(precstr2 == "sgs")
			ptype = SGS;
		else if(precstr2 == "ilu0")
			ptype = ILU0;
		else {
			printf("BLASTed: Preconditioner type not available!\n");
			abort();
		}
	}
	
	PetscInt sweeps[2];
	if(ptype == SGS || ptype == ILU0)
	{
		PetscOptionsHasName(NULL, NULL, "-blasted_async_sweeps", &set);
		if(set == PETSC_FALSE) {
			printf("BLASTed: Number of async sweeps not set!\n");
			abort();
		}
		else {
			PetscBool flag = PETSC_FALSE;
			PetscInt nmax;
			PetscOptionsGetIntArray(NULL, NULL, "-blasted_async_sweeps", sweeps, &nmax, &flag);
			
			if(flag == PETSC_FALSE || nmax < 2) {
				printf("BLASTed: Number of async sweeps not set properly!\n");
				abort();
			}
		}
	}
	else {
		sweeps[0] = 1; sweeps[1] = 1;
	}

	//std::printf("BLASTed: newDataFromOptions: Setting up preconditioner with\n");
	//std::printf("ptype = %d and sweeps = %d,%d.\n", ptype, sweeps[0], sweeps[1]);

	ctx->bmat = nullptr; 
	ctx->prectype = ptype;
	ctx->nbuildsweeps = sweeps[0]; 
	ctx->napplysweeps = sweeps[1];
	ctx->first_setup_done = true;
	ctx->cputime = ctx->walltime = ctx->factorcputime = ctx->factorwalltime =
		ctx->applycputime = ctx->applywalltime = 0;
	
	if(ctx->prectype == JACOBI)
		ierr = PCShellSetName(pc, "Blasted-Jacobi");
	else if(ctx->prectype == SGS)
		ierr = PCShellSetName(pc, "Blasted-SGS");
	else
		ierr = PCShellSetName(pc, "Blasted-ILU0");

	return ierr;
}

/** \brief Generates a matrix view from the preconditioning operator in a sub PC
 *
 * \warning We assume that the pc passed here is a subpc, ie, a local preconditioner.
 * \todo TODO: Implement this for block preconditioners as well
 * \param[in,out] pc PETSc preconditioner context
 */
PetscErrorCode createNewBlockMatrixView(PC pc)
{
	PetscErrorCode ierr = 0;

	PetscInt firstrow, lastrow, localrows, localcols, globalrows, globalcols;
	/*PetscInt numprocs;
	MPI_Comm_size(PETSC_COMM_WORLD,&numprocs);*/
	
	// get control structure
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);

	// delete old matrix
	BlastedPetscMat* op = reinterpret_cast<BlastedPetscMat*>(ctx->bmat);
	delete op;

	/* get the local preconditioning matrix
	 * we operate on the diagonal matrix block corresponding to this process
	 */
	Mat A;
	ierr = PCGetOperators(pc, NULL, &A); CHKERRQ(ierr);
	// Petsc distributes matrices by row
	ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); CHKERRQ(ierr);	
	ierr = MatGetLocalSize(A, &localrows, &localcols); CHKERRQ(ierr);
	ierr = MatGetSize(A, &globalrows, &globalcols); CHKERRQ(ierr);
	const Mat_SeqAIJ *Adiag;
	//const Mat_SeqAIJ *Aoffdiag;

	// get access to local matrix entries

	/*if(numprocs > 1) {
		const Mat_MPIAIJ *Atemp = (const Mat_MPIAIJ*)A->data;
		
		// Square diagonal portion of the local part of the matrix
		Adiag = (const Mat_SeqAIJ*)Atemp->A->data;	
	
		// Off-diagonal portion of the local part of the matrix
		//Aoffdiag = (const Mat_SeqAIJ*)Atemp->B->data;
	}
	else {*/
		Adiag = (const Mat_SeqAIJ*)A->data;
		//Aoffdiag = NULL;
	//}

#if 0
	const PetscInt Adnnz = Adiag->nz;

	printf("BLASTed: createNewBlockMatrix(): firstrow = %d, lastrow = %d,\n",
			firstrow, lastrow);
	printf("     localrows = %d, localcols = %d, globalrows = %d, globalcols = %d\n", 
			localrows, localcols, globalrows, globalcols);
	
	if(localrows != localcols)
		printf("! BLASTed: createNewBlockMatrix(): Local matrix is not square! Size is %dx%d.\n", 
				localrows, localcols);
	
	if(localrows != (lastrow-firstrow))
		printf("! BLASTed: createNewBlockMatrix(): \
				Ownership range %d and local rows %d are not consistent with each other!", 
				lastrow-firstrow, localrows);
	
	if(Adnnz != Adiag->i[localrows])
		printf("! BLASTed: createNewBlockMatrix(): M nnz = %d, last entry of M rowp = %d!\n", 
				Adnnz, Adiag->i[localrows]);
#endif

	switch(ctx->bs) {
		case 0:
			printf("BLASTed: createNewBlockMatrix: Invalid block size 0!\n");
			abort();
			break;
		case 1:
			op = new BSRMatrixView<PetscReal, PetscInt,1>(localrows, 
					Adiag->i, Adiag->j, Adiag->a, Adiag->diag,
					ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		/*case 2:
			op = new BSRMatrix<PetscReal,PetscInt,2>(localrows, Adrowp, Adcols, Advals, Addiagind,
					ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		case 3:
			op = new BSRMatrix<PetscReal,PetscInt,3>(localrows, Adrowp, Adcols, Advals, Addiagind,
					ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		case 4:
			op = new BSRMatrix<PetscReal,PetscInt,4>(localrows, Adrowp, Adcols, Advals, Addiagind,
					ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		case 5:
			op = new BSRMatrix<PetscReal,PetscInt,5>(localrows, Adrowp, Adcols, Advals, Addiagind,
					ctx->nbuildsweeps,ctx->napplysweeps);
			break;*/
		default:
			printf("BLASTed: createNewBlockMatrix: That block size is not supported!\n");
			abort();
	}

	ctx->bmat = reinterpret_cast<void*>(op);
	return ierr;
}

extern "C" {

PetscErrorCode cleanup_blasted(PC pc)
{
	PetscErrorCode ierr = 0;
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);
	
	BlastedPetscMat* mat = reinterpret_cast<BlastedPetscMat*>(ctx->bmat);
	delete mat;

	return ierr;
}

PetscErrorCode compute_preconditioner_blasted(PC pc)
{
	PetscErrorCode ierr = 0;
	
	// get control structure
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);

	if(!ctx->first_setup_done) {
		ierr = setupDataFromOptions(pc);
		CHKERRQ(ierr);
	}

	ierr = createNewBlockMatrixView(pc); CHKERRQ(ierr);
	BlastedPetscMat *const op = reinterpret_cast<BlastedPetscMat*>(ctx->bmat);

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	// setup preconditioners
	switch(ctx->prectype) {
		case JACOBI:
			op->precJacobiSetup();
			break;
		case SGS:
			op->precSGSSetup();
			break;
		case ILU0:
			op->precILUSetup();
			break;
		default:
			printf("BLASTed: compute_preconditioner: Preconditioner not available!\n");
			ierr = -1;
	}
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	ctx->factorwalltime += (finalwtime - initialwtime);
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	ctx->factorcputime += (finalctime - initialctime);

	return ierr;
}

PetscErrorCode apply_local_blasted(PC pc, Vec r, Vec z)
{
	PetscErrorCode ierr = 0;
	const PetscReal *ra;
	PetscReal *za;
	PetscInt start, end;
	VecGetOwnershipRange(r, &start, &end);

	Blasted_data* ctx;
	PCShellGetContext(pc, (void**)&ctx);
	const BlastedPetscMat *const mat = reinterpret_cast<const BlastedPetscMat *>(ctx->bmat);

#ifdef DEBUG
	if(mat->dim() != end-start) {
		printf("! apply_local: Dimension of the input vector r\n");
		printf("     does not match dimension of the preconditioning matrix!\n");
		ierr = -1;
	}
#endif
	
	VecGetArray(z, &za);
	VecGetArrayRead(r, &ra);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	switch(ctx->prectype) {
		case JACOBI:
			mat->precJacobiApply(ra, za);
			break;
		case SGS:
			mat->precSGSApply(ra, za);
			break;
		case ILU0:
			mat->precILUApply(ra, za);
			break;
		default:
			printf("BLASTed: apply_local: Preconditioner not available!\n");
			ierr = -1;
	}

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	ctx->applywalltime += (finalwtime - initialwtime);
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	ctx->applycputime += (finalctime - initialctime);

	VecRestoreArrayRead(r, &ra);
	VecRestoreArray(z, &za);
	
	return ierr;
}

PetscErrorCode get_blasted_timing_data(PC pc, double *const factorcputime, 
		double *const factorwalltime, double *const applycputime, double *const applywalltime)
{
	Blasted_data* ctx;
	PetscErrorCode ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);
	*factorcputime = ctx->factorcputime;
	*factorwalltime = ctx->factorwalltime;
	*applycputime = ctx->applycputime;
	*applywalltime = ctx->applywalltime;
	return ierr;
}

}
