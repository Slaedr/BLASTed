/** \file blasted_petsc.cpp
 * \brief C wrapper for using BLASTed functionality with PETSc
 * \author Aditya Kashi
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/time.h>

#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

#include <blockmatrices.hpp>
#include <blasted_petsc.h>

using namespace blasted;

typedef SRMatrixView<PetscReal, PetscInt> BlastedPetscMat;

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
		else if(precstr2 == "sapilu0")
			ptype = SAPILU0;
		else {
			printf("BLASTed: Preconditioner type not available!\n");
			abort();
		}
	}
	
	PetscInt sweeps[2];
	if(ptype != JACOBI)
	{
		PetscOptionsHasName(NULL, NULL, "-blasted_async_sweeps", &set);
		if(set == PETSC_FALSE) {
			printf("BLASTed: Number of async sweeps not set!\n");
			abort();
		}
		else {
			PetscBool flag = PETSC_FALSE;
			PetscInt nmax = 2;
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

#ifdef DEBUG
	std::printf("BLASTed: newDataFromOptions: Setting up preconditioner with\n");
	std::printf("ptype = %d and sweeps = %d,%d.\n", ptype, sweeps[0], sweeps[1]);
#endif

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
	else if(ctx->prectype == ILU0)
		ierr = PCShellSetName(pc, "Blasted-ILU0");
	else if(ctx->prectype == SAPILU0)
		ierr = PCShellSetName(pc, "Blasted-SeqApILU0");

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
	ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); CHKERRQ(ierr);	
	ierr = MatGetLocalSize(A, &localrows, &localcols); CHKERRQ(ierr);
	ierr = MatGetSize(A, &globalrows, &globalcols); CHKERRQ(ierr);
	assert(localrows == localcols);
	assert(globalrows == globalcols);

	// get access to local matrix entries
	const Mat_SeqAIJ *const Adiag = (const Mat_SeqAIJ*)A->data;
	const Mat_SeqBAIJ *const Abdiag = (const Mat_SeqBAIJ*)A->data;
	
	// ensure diagonal entry locations have been computed; this is necessary for BAIJ matrices
	// as a bonus, check for singular diagonals
	PetscBool diagmissing = PETSC_FALSE;
	PetscInt badrow = -1;
	ierr = MatMissingDiagonal(A, &diagmissing, &badrow); CHKERRQ(ierr);
	if(diagmissing == PETSC_TRUE) {
		SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "! Zero diagonal in (block-)row %d!", badrow);
	}

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
				(localrows/ctx->bs, Abdiag->i, Abdiag->j, Abdiag->a, Abdiag->diag, 
				 ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		case 4:
			op = new BSRMatrixView<PetscReal,PetscInt,4,Eigen::ColMajor>
				(localrows/ctx->bs, Abdiag->i, Abdiag->j, Abdiag->a, Abdiag->diag, 
				 ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		case 5:
			op = new BSRMatrixView<PetscReal,PetscInt,5,Eigen::ColMajor>
				(localrows/ctx->bs, Abdiag->i, Abdiag->j, Abdiag->a, Abdiag->diag, 
				 ctx->nbuildsweeps,ctx->napplysweeps);
			break;
		default:
			printf("BLASTed: createNewBlockMatrix: That block size is not supported!\n");
			abort();
	}

	ctx->bmat = reinterpret_cast<void*>(op);
	return ierr;
}

PetscErrorCode updateBlockMatrixView(PC pc)
{
	PetscErrorCode ierr = 0;
	PetscInt firstrow, lastrow, localrows, localcols, globalrows, globalcols;
	
	// get control structure
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);
	BlastedPetscMat* op = reinterpret_cast<BlastedPetscMat*>(ctx->bmat);

	/* get the local preconditioning matrix
	 * we operate on the diagonal matrix block corresponding to this process
	 */
	Mat A;
	ierr = PCGetOperators(pc, NULL, &A); CHKERRQ(ierr);
	ierr = MatGetOwnershipRange(A, &firstrow, &lastrow); CHKERRQ(ierr);	
	ierr = MatGetLocalSize(A, &localrows, &localcols); CHKERRQ(ierr);
	ierr = MatGetSize(A, &globalrows, &globalcols); CHKERRQ(ierr);
	assert(localrows == localcols);
	assert(globalrows == globalcols);

	// get access to local matrix entries
	const Mat_SeqAIJ *const Adiag = (const Mat_SeqAIJ*)A->data;
	const Mat_SeqBAIJ *const Abdiag = (const Mat_SeqBAIJ*)A->data;

	if(ctx->bs <= 0 || ctx->bs > 5 || ctx->bs == 2)
		SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "BLASTed: Block size %d is not supported!", ctx->bs);

	if(ctx->bs == 1)
		op->wrap(localrows/ctx->bs, Adiag->i, Adiag->j, Adiag->a, Adiag->diag);
	else
		op->wrap(localrows/ctx->bs, Abdiag->i, Abdiag->j, Abdiag->a, Abdiag->diag);

	ctx->bmat = reinterpret_cast<void*>(op);
	return ierr;
}

extern "C" {

Blasted_data_list newBlastedDataList()
{
	Blasted_data_list b;
	b.ctxlist = NULL;
	b.size = 0;
	b.factorcputime = b.factorwalltime = b.applycputime = b.applywalltime = 0.0;
	return b;
}

void destroyBlastedDataList(Blasted_data_list *const b)
{
	while(b->ctxlist != NULL) {
		Blasted_data *temp = b->ctxlist;
		b->ctxlist = b->ctxlist->next;
		delete temp;
		b->size--;
	}

	if(b->size != 0)
		throw std::logic_error("Could not delete Blasted_data_list properly!");
}

Blasted_data newBlastedDataContext()
{
	Blasted_data ctx;
	ctx.bmat = NULL;
	ctx.first_setup_done = false;
	ctx.cputime = ctx.walltime = ctx.factorcputime = ctx.factorwalltime
		= ctx.applycputime = ctx.applywalltime = 0.0;
	ctx.next = NULL;
	return ctx;
}

void appendBlastedDataContext(Blasted_data_list *const bdl, const Blasted_data bd)
{
	Blasted_data *node = new Blasted_data;
	*node = bd;

	node->next = bdl->ctxlist;
	bdl->ctxlist = node;

	bdl->size++;
}

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
		ierr = setupDataFromOptions(pc); CHKERRQ(ierr);
		ierr = createNewBlockMatrixView(pc); CHKERRQ(ierr);
	}

	ierr = updateBlockMatrixView(pc); CHKERRQ(ierr);

	BlastedPetscMat *const op = reinterpret_cast<BlastedPetscMat*>(ctx->bmat);

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;
	
	// setup preconditioners
	switch(ctx->prectype) {
		case SGS:
			op->precSGSSetup();
			break;
		case ILU0:
		case SAPILU0:
			op->precILUSetup();
			break;
		case JACOBI:
			op->precJacobiSetup();
			break;
		default:
			SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Preconditioner not available!");
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
		SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, 
				"BLASTed: Dimension of vector does not match that of the matrix- %d vs %d!\n", 
				mat->dim(), end-start);
	}
#endif
	
	VecGetArray(z, &za);
	VecGetArrayRead(r, &ra);
	
	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	switch(ctx->prectype) {
		case SGS:
			mat->precSGSApply(ra, za);
			break;
		case ILU0:
			mat->precILUApply(ra, za);
			break;
		case SAPILU0:
			mat->precILUApply_seq(ra, za);
			break;
		case JACOBI:
			mat->precJacobiApply(ra, za);
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

PetscErrorCode setup_blasted_stack(KSP ksp, Blasted_data_list *const bctv, const int ictx)
{
	PetscErrorCode ierr = 0;
	PC pc;
	ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
	PetscBool isbjacobi, isasm, isshell, ismg, isgamg, isksp;
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCBJACOBI,&isbjacobi); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCASM,&isasm); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&isshell); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCMG,&ismg); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCGAMG,&isgamg); CHKERRQ(ierr);
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCKSP,&isksp); CHKERRQ(ierr);

	if(isbjacobi || isasm)
	{
		PetscInt nlocalblocks, firstlocalblock;
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		KSP *subksp;
		if(isbjacobi) {
			ierr = PCBJacobiGetSubKSP(pc, &nlocalblocks, &firstlocalblock, &subksp); CHKERRQ(ierr);
		}
		else {
			ierr = PCASMGetSubKSP(pc, &nlocalblocks, &firstlocalblock, &subksp); CHKERRQ(ierr);
		}
		if(nlocalblocks != 1)
			SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, 
					"Only one subdomain per rank is supported.");

		ierr = setup_blasted_stack(subksp[0], bctv, ictx); CHKERRQ(ierr);
	}
	else if(ismg || isgamg) {
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		PetscInt nlevels;
		ierr = PCMGGetLevels(pc, &nlevels); CHKERRQ(ierr);

		for(int ilvl = 1; ilvl < nlevels; ilvl++) {
			KSP smootherctx;
			ierr = PCMGGetSmoother(pc, ilvl , &smootherctx); CHKERRQ(ierr);
			ierr = setup_blasted_stack(smootherctx, bctv, ictx); CHKERRQ(ierr);
			/*
			KSP smootherup, smootherdown;
			ierr = PCMGGetSmootherDown(pc, ilvl , &smootherdown); CHKERRQ(ierr);
			ierr = PCMGGetSmootherUp(pc, ilvl , &smootherup); CHKERRQ(ierr);
			ierr = setup_blasted_stack(smootherdown, bctv, ilvl); CHKERRQ(ierr);
			ierr = setup_blasted_stack(smootherup, bctv, ilvl+nlevels-1); CHKERRQ(ierr);*/
		}
		KSP coarsesolver;
		ierr = PCMGGetCoarseSolve(pc, &coarsesolver); CHKERRQ(ierr);
		ierr = setup_blasted_stack(coarsesolver, bctv, ictx); CHKERRQ(ierr);
	}
	else if(isksp) {
		ierr = KSPSetUp(ksp); CHKERRQ(ierr); 
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		KSP subksp;
		ierr = PCKSPGetKSP(pc, &subksp); CHKERRQ(ierr);
		ierr = setup_blasted_stack(subksp, bctv, ictx); CHKERRQ(ierr);
	}
	else if(isshell) {
		// if the PC is shell, this is the relevant KSP to pass to BLASTed for setup
		std::cout << "setup_blasted_stack(): Found valid parent KSP for BLASTed.\n";

		appendBlastedDataContext(bctv, newBlastedDataContext());
		// The new context is appended to the head of the list,
		// so we setup the BLASTed preconditioner using the context at the head of the list
		ierr = setup_localpreconditioner_blasted(ksp, bctv->ctxlist); CHKERRQ(ierr);
	}

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
	PetscInt matbs; MatType mtype;
	ierr = MatGetBlockSize(A, &matbs); CHKERRQ(ierr);
	ierr = MatGetType(A, &mtype); CHKERRQ(ierr);
	bool isBlockMat = false, islocal = false;
	if(!strcmp(mtype, MATBAIJ) || !strcmp(mtype,MATMPIBAIJ) || !strcmp(mtype,MATSEQBAIJ) ||
		!strcmp(mtype, MATBAIJMKL) || !strcmp(mtype,MATMPIBAIJMKL) || !strcmp(mtype,MATSEQBAIJMKL) )
	{
		isBlockMat = true;
	}
	if(!strcmp(mtype, MATSEQAIJ) || !strcmp(mtype, MATSEQBAIJ) || !strcmp(mtype, MATSEQAIJMKL)
			|| !strcmp(mtype, MATSEQBAIJMKL))
		islocal = true;

	PC pc, subpc;

	KSPGetPC(ksp, &pc);
	PetscBool isshell;
	ierr = PetscObjectTypeCompare((PetscObject)pc,PCSHELL,&isshell); CHKERRQ(ierr);

	// extract sub pc
	if(isshell) {
		subpc = pc;
		// only for single-process runs
		if(!islocal)
			SETERRQ(comm, PETSC_ERR_SUP, 
					"PC as PCSHELL is only supported for local solvers.");
	}
	else {
		SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "Need SHELL preconditioner for BLASTed!\n");
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

void computeTotalTimes(Blasted_data_list *const bctv)
{
	bctv->factorcputime = bctv->factorwalltime = bctv->applycputime = bctv->applywalltime = 0.0;

	Blasted_data *node = bctv->ctxlist;
	while(node != NULL){
		bctv->factorwalltime += node->factorwalltime;
		bctv->applywalltime += node->applywalltime;
		bctv->factorcputime += node->factorcputime;
		bctv->applycputime += node->applycputime;
		node = node->next;
	}
}

}
