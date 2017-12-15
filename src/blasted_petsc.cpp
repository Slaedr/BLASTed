/** \file blasted_petsc.cpp
 * \brief C wrapper for using BLASTed functionality with PETSc
 * \author Aditya Kashi
 */

#include <cmath>
#include <cstdio>
//#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/time.h>

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#include <blockmatrices.hpp>
#include <blasted_petsc.h>

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
	assert(localrows == localcols);
	assert(globalrows == globalcols);

	// get access to local matrix entries
	//const Mat_SeqAIJ *Aoffdiag;
	const Mat_SeqAIJ *const Adiag = (const Mat_SeqAIJ*)A->data;

	switch(ctx->bs) {
		case 0:
			printf("BLASTed: createNewBlockMatrix: Invalid block size 0!\n");
			abort();
			break;
		case 1:
			op = new CSRMatrixView<PetscReal, PetscInt>
				(localrows, Adiag->i, Adiag->j, Adiag->a, Adiag->diag,
				ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		/*case 2:
			op = new BSRMatrixView<PetscReal,PetscInt,2,Eigen::ColMajor>
				(localrows/bs, Adiag->i, Adiag->j, Adiag->a, Adiag->diag, 
				 ctx->nbuildsweeps,ctx->napplysweeps);*/
		case 3:
			op = new BSRMatrixView<PetscReal,PetscInt,3,Eigen::ColMajor>
				(localrows/ctx->bs, Adiag->i, Adiag->j, Adiag->a, Adiag->diag, 
				 ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		case 4:
			op = new BSRMatrixView<PetscReal,PetscInt,4,Eigen::ColMajor>
				(localrows/ctx->bs, Adiag->i, Adiag->j, Adiag->a, Adiag->diag, 
				 ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		case 5:
			op = new BSRMatrixView<PetscReal,PetscInt,5,Eigen::ColMajor>
				(localrows/ctx->bs, Adiag->i, Adiag->j, Adiag->a, Adiag->diag, 
				 ctx->nbuildsweeps,ctx->napplysweeps);
			break;
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

PetscErrorCode setup_localpreconditioner_blasted(KSP ksp, Blasted_data *const bctx)
{
	PetscErrorCode ierr = 0;
	int mpisize, mpirank;
	MPI_Comm comm = PETSC_COMM_WORLD;
	MPI_Comm_size(comm,&mpisize);
	MPI_Comm_rank(comm,&mpirank);

	Mat A;
	ierr = KSPGetOperators(ksp, NULL, &A); CHKERRQ(ierr);
	PetscInt m,n;
	ierr = MatGetLocalSize(A, &m, &n); CHKERRQ(ierr);
	PetscInt matbs; MatType mtype;
	ierr = MatGetBlockSize(A, &matbs); CHKERRQ(ierr);
	ierr = MatGetType(A, &mtype); CHKERRQ(ierr);
	//printf(" Rank %d: Local size: %d, %d. Block size = %d.\n", rank, m, n, matbs);
	bool isBlockMat = false;
	if(!strcmp(mtype, MATBAIJ) || !strcmp(mtype,MATMPIBAIJ) || !strcmp(mtype,MATSEQBAIJ)) {
		isBlockMat = true;
	}

	KSP *subksp;
	PC pc, subpc;

	KSPGetPC(ksp, &pc);
	PetscBool isbjacobi, isasm, isshell;
	PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&isbjacobi);
	PetscObjectTypeCompare((PetscObject)pc,PCASM,&isasm);
	PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&isshell);
	if(isbjacobi)
	{
		// extract sub pc
		PetscInt nlocalblocks, firstlocalblock;
		KSPSetUp(ksp); PCSetUp(pc);
		ierr = PCBJacobiGetSubKSP(pc, &nlocalblocks, &firstlocalblock, &subksp);
		CHKERRQ(ierr);
		assert(nlocalblocks == 1);
		KSPGetPC(subksp[0], &subpc);
	}
	else if(isasm)
	{
		// extract sub pc
		PetscInt nlocalblocks, firstlocalblock;
		KSPSetUp(ksp); PCSetUp(pc);
		ierr = PCASMGetSubKSP(pc, &nlocalblocks, &firstlocalblock, &subksp);
		CHKERRQ(ierr);
		assert(nlocalblocks == 1);
		KSPGetPC(subksp[0], &subpc);
	}
	else if(isshell) {
		subpc = pc;
		// only for single-process runs
		assert(mpisize == 1);
	}
	else {
		SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "Invalid global preconditioner for BLASTed!\n");
	}

	// setup the PC
	bctx->bs = isBlockMat ? matbs : 1; 
	bctx->first_setup_done = false;
	ierr = PCShellSetContext(subpc, (void*)bctx);                   CHKERRQ(ierr);
	ierr = PCShellSetSetUp(subpc, &compute_preconditioner_blasted); CHKERRQ(ierr);
	ierr = PCShellSetApply(subpc, &apply_local_blasted);            CHKERRQ(ierr);
	ierr = PCShellSetDestroy(subpc, &cleanup_blasted);              CHKERRQ(ierr);

	return ierr;
}

}
