/** \file blasted_petsc.cpp
 * \brief C wrapper for using BLASTed functionality with PETSc
 * \author Aditya Kashi
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#include "blockmatrices.hpp"

#include "blasted_petsc.h"

using namespace blasted;

typedef LinearOperator<PetscReal, PetscInt> BlastedPetscMat;

#define PETSCOPTION_STR_LEN 10

/// Returns a dynamically allocated Blasted_data object after setting options from PETSc options
static Blasted_data* newDataFromOptions(const int bsize)
{
	Prec_type ptype;

	PetscBool set = PETSC_FALSE;
	PetscOptionsHasName(NULL, NULL, "blasted_pc_type", &set);
	if(set == PETSC_FALSE) {
		printf("BLASTed: Preconditioner type not set! Setting to Jacobi.\n");
		ptype = JACOBI;
	}
	else {
		char precstr[PETSCOPTION_STR_LEN];
		PetscBool flag = PETSC_FALSE;
		PetscOptionsGetString(NULL, NULL, "blasted_pc_type", 
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
	PetscOptionsHasName(NULL, NULL, "blasted_async_sweeps", &set);
	if(set == PETSC_FALSE) {
		printf("BLASTed: Number of async sweeps not set!\n");
		abort();
	}
	else {
		PetscBool flag = PETSC_FALSE;
		PetscInt nmax;
		PetscOptionsGetIntArray(NULL, NULL, "blasted_async_sweeps", sweeps, &nmax, &flag);
		
		if(flag == PETSC_FALSE || nmax < 2) {
			printf("BLASTed: Number of async sweeps not set properly!\n");
			abort();
		}
	}

	/*Blasted_data* ctx = new Blasted_data {bmat=NULL; bs = bsize; prectype = ptype;
		nbuildsweeps = sweeps[0]; napplysweeps = sweeps[1];
		cputime = 0; walltime = 0; factorcputime = 0; factorwalltime = 0;
		applycputime = 0; applywalltime = 0;
	}*/
	Blasted_data* ctx = new Blasted_data {nullptr, bsize, ptype,
		sweeps[0], sweeps[1],
		0, 0, 0, 0, 0, 0
	};

	return ctx;
}

/** \brief Generates a new native matrix from the preconditioning operator in a PC
 *
 * \param[in,out] pc PETSc preconditioner context
 */
void createNewBlockMatrix(PC pc)
{
	PetscInt firstrow, lastrow, localrows, localcols, globalrows, globalcols, numprocs;
	MPI_Comm_size(PETSC_COMM_WORLD,&numprocs);
	
	// get control structure
	Blasted_data* ctx;
	PCShellGetContext(pc, (void**)&ctx);

	// delete old matrix
	const BlastedPetscMat* op = (const BlastedPetscMat*)ctx->bmat;
	delete op;

	/* get the local preconditioning matrix
	 * we operate on the diagonal matrix block corresponding to this process
	 */
	Mat A;
	PCGetOperators(pc, NULL, &A);
	// Petsc distributes matrices by row
	MatGetOwnershipRange(A, &firstrow, &lastrow);		
	MatGetLocalSize(A, &localrows, &localcols);
	MatGetSize(A, &globalrows, &globalcols);
	const Mat_SeqAIJ *Adiag;
	//const Mat_SeqAIJ *Aoffdiag;

	// get access to local matrix entries

	if(numprocs > 1) {	
		const Mat_MPIAIJ *Atemp = (const Mat_MPIAIJ*)A->data;
		
		// Square diagonal portion of the local part of the matrix
		Adiag = (const Mat_SeqAIJ*)Atemp->A->data;	
	
		// Off-diagonal portion of the local part of the matrix
		//Aoffdiag = (const Mat_SeqAIJ*)Atemp->B->data;
	}
	else {
		Adiag = (const Mat_SeqAIJ*)A->data;
		//Aoffdiag = NULL;
	}

	const PetscInt* Adrowp = Adiag->i;
	const PetscInt* Adcols = Adiag->j;
	const PetscInt* Addiagind = Adiag->diag;
	const PetscReal* Advals = Adiag->a;
	const PetscInt Adnnz = Adiag->nz;

#ifdef DEBUG
	printf("BLASTed: createNewBlockMatrix(): firstrow = %d, lastrow = %d, \
			localrows = %d, localcols = %d, globalrows = %d, globalcols = %d\n", 
			firstrow, lastrow, localrows, localcols, globalrows, globalcols);
	
	if(localrows != localcols)
		printf("! BLASTed: createNewBlockMatrix(): Local matrix is not square! Size is %dx%d.\n", 
				localrows, localcols);
	
	if(localrows != (lastrow-firstrow))
		printf("! BLASTed: createNewBlockMatrix(): \
				Ownership range %d and local rows %d are not consistent with each other!", 
				lastrow-firstrow, localrows);
	
	if(Adnnz != Adrowp[localrows])
		printf("! BLASTed: createNewBlockMatrix(): M nnz = %d, last entry of M rowp = %d!\n", 
				Adnnz, Adrowp[localrows]);
#endif

	switch(ctx->bs) {
		case 0:
			printf("BLASTed: createNewBlockMatrix: Invalid block size 0!\n");
			abort();
			break;
		case 1:
			op = new BSRMatrix<PetscReal, PetscInt,1>(localrows, 
					Adrowp, Adcols, Advals, Addiagind,
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

	ctx->bmat = (void*)op;
}

extern "C" {

PetscErrorCode setup_blasted(PC pc, const int bs)
{
	Blasted_data* ctx = newDataFromOptions(bs);
	PetscErrorCode ierr = PCShellSetContext(pc, (void*)ctx); CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode cleanup_blasted(PC pc)
{
	PetscErrorCode ierr = 0;
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);
	
	BlastedPetscMat* mat = (BlastedPetscMat*)ctx->bmat;
	delete mat;
	delete ctx;

	return ierr;
}

PetscErrorCode compute_preconditioner(PC pc)
{
	PetscErrorCode ierr = 0;

	createNewBlockMatrix(pc);
	
	// get control structure
	Blasted_data* ctx;
	PCShellGetContext(pc, (void**)&ctx);
	BlastedPetscMat *const op = (BlastedPetscMat *const)ctx->bmat;

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
	}
	
	
	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	ctx->factorwalltime += (finalwtime - initialwtime);
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	ctx->factorcputime += (finalctime - initialctime);

	return ierr;
}

PetscErrorCode apply_local(PC pc, Vec r, Vec z)
{
	const PetscReal *ra;
	PetscReal *za;
	PetscInt start, end;
	VecGetOwnershipRange(r, &start, &end);

	Blasted_data* ctx;
	PCShellGetContext(pc, (void**)&ctx);
	const BlastedPetscMat *const mat = (const BlastedPetscMat *const)ctx->bmat;

#ifdef DEBUG
	if(mat->dim() != end-start) {
		printf("! apply_fgpilu_jacobi_local: Dimension of the input vector r\n");
		printf("     does not match dimension of the preconditioning matrix!\n");
		return -1;
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
	}

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	ctx->applywalltime += (finalwtime - initialwtime);
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	ctx->applycputime += (finalctime - initialctime);

	VecRestoreArrayRead(r, &ra);
	VecRestoreArray(z, &za);
	
	return 0;
}

}
