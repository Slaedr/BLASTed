/** \file blasted_petsc.cpp
 * \brief C wrapper for using BLASTed functionality with PETSc
 * \author Aditya Kashi
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sys/time.h>
#include <omp.h>

#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

#include "solvertypes.h"
#include "solverops_jacobi.hpp"
#include "solverops_sgs.hpp"
#include "solverops_ilu0.hpp"
#include "solverfactory.hpp"
#include "blasted_petsc_ext.hpp"
#include "preconditioner_diagnostics.hpp"

using namespace blasted;

typedef SRPreconditioner<PetscReal,PetscInt> BlastedPreconditioner;


/// Reads an int option from the PETSc options database
static int get_int_petscoptions(const char *const option_tag)
{
	PetscBool set = PETSC_FALSE;
	int val = 0;
	PetscOptionsGetInt(NULL, NULL, option_tag, &val, &set);
	if(!set) {
		printf("BLASTed: %s not set!\n", option_tag);
		abort();
	}
	return val;
}

/// Reads an optional bool option from the PETSc options database
static int get_optional_bool_petscoptions(const char *const option_tag, const bool default_value)
{
	PetscBool set = PETSC_FALSE;
	PetscBool val = default_value == true ? PETSC_TRUE : PETSC_FALSE;
	int ierr = PetscOptionsGetBool(NULL, NULL, option_tag, &val, &set);
	if(ierr) {
		throw std::runtime_error("Petsc could not get optional bool option!");
	}
	if(!set) {
		printf(" BLASTed: %s not set; using default value of %d\n", option_tag, default_value);
	}
	return (val == PETSC_TRUE ? true : false);
}

/// Reads a compulsory bool option from the PETSc options database
static int get_bool_petscoptions(const char *const option_tag)
{
	PetscBool set = PETSC_FALSE;
	PetscBool val;
	int ierr = PetscOptionsGetBool(NULL, NULL, option_tag, &val, &set);
	if(ierr) {
		throw std::runtime_error("Petsc could not get optional bool option!");
	}
	if(!set) {
		throw std::runtime_error("Bool option " + std::string(option_tag) + " not set!");
	}
	return (val == PETSC_TRUE ? true : false);
}

/// Read a string option from the PETSc options database
/** \note Aborts the program if the option was not found
 */
static void get_string_petscoptions(const char *const option_tag, char outstr[BLASTED_OPT_STRLEN])
{
	// Prec type
	PetscBool flag = PETSC_FALSE;
	PetscOptionsGetString(NULL, NULL, option_tag, outstr, BLASTED_OPT_STRLEN, &flag);
	if(flag == PETSC_FALSE) {
		printf("BLASTed: %s not set!\n", option_tag); fflush(stdout);
		abort();
	}
}

/// Check if sequential factorization or application was requested and set build/apply sweeps
static void setSweeps_checkSeq(const Blasted_data *const ctx, AsyncSolverSettings& settings)
{
	static_assert(BLASTED_SEQUENTIAL_SYMBOL < 0, "Symbol of sequential build/apply must be -ve!");

	settings.nbuildsweeps = ctx->nbuildsweeps;
	settings.napplysweeps = ctx->napplysweeps;

	if(settings.prectype == BLASTED_SEQILU0)
		return;

	// Cases where both build and apply are sequential
	if((ctx->napplysweeps == BLASTED_SEQUENTIAL_SYMBOL && ctx->nbuildsweeps == BLASTED_SEQUENTIAL_SYMBOL)
	   || (ctx->napplysweeps == BLASTED_SEQUENTIAL_SYMBOL && settings.prectype == BLASTED_SFILU0)
	   || (ctx->nbuildsweeps == BLASTED_SEQUENTIAL_SYMBOL && settings.prectype == BLASTED_SAPILU0))
	{
		const int nthreads = omp_get_max_threads();
		if(nthreads != 1) {
			std::cout << "WARNING: Both sequential build and apply are requested, but " << nthreads
			          << " threads are being used!" << std::endl;
			std::cout << " Sequential preconditioner will be used for this run\n";
		}
		settings.prectype = BLASTED_SEQILU0;
		settings.nbuildsweeps = 1;
		settings.napplysweeps = 1;
		return;
	}

	if(ctx->napplysweeps == BLASTED_SEQUENTIAL_SYMBOL)
	{
		if(settings.prectype != BLASTED_ILU0 && settings.prectype != BLASTED_SAPILU0)
		   //&& settings.prectype != BLASTED_SFILU0)
			throw std::runtime_error(" Seq. appl. only supported with async ILU factorization!");
		settings.napplysweeps = 1;
		settings.prectype = BLASTED_SAPILU0;
		printf("  Sequential application requested.\n");
	}
	if(ctx->nbuildsweeps == BLASTED_SEQUENTIAL_SYMBOL)
	{
		if(settings.prectype != BLASTED_ILU0 && settings.prectype != BLASTED_SFILU0)
			//&& settings.prectype != BLASTED_SAPILU0)
			throw std::runtime_error(" Seq. fact. only supported with async triangular application!");
		settings.nbuildsweeps = 1;
		settings.prectype = BLASTED_SFILU0;
		printf("  Sequential factorization requested.\n");
	}
}

/// Sets options from PETSc options
static PetscErrorCode setupDataFromOptions(PC pc)
{
	PetscErrorCode ierr=0;
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);

	const FactoryBase<PetscReal,PetscInt> *const factory
		= (const FactoryBase<PetscReal,PetscInt>*)ctx->bfactory;

	// Prec type
	get_string_petscoptions("-blasted_pc_type", ctx->prectypestr);
	const BlastedSolverType ptype = factory->solverTypeFromString(ctx->prectypestr);

	PetscInt sweeps[2];

	if(ptype != BLASTED_JACOBI && ptype != BLASTED_LEVEL_SGS && ptype != BLASTED_NO_PREC)
	{
		// Params for async iterations

		// sweeps
		PetscBool flag = PETSC_FALSE;
		PetscInt nmax = 2;
		PetscOptionsGetIntArray(NULL, NULL, "-blasted_async_sweeps", sweeps, &nmax, &flag);

		if(flag == PETSC_FALSE || nmax < 2) {
			printf("BLASTed: Number of async sweeps not set properly!\n"); fflush(stdout);
			abort();
		}

		char applyitertype[100];
		get_string_petscoptions("-blasted_pc_apply_iter_type", applyitertype);
		ctx->aplitertype = getIterTypeFromString(applyitertype);

		if(ptype == BLASTED_ILU0 || ptype == BLASTED_SAPILU0 || ptype == BLASTED_ASYNC_LEVEL_ILU0)
		{
			ctx->scale = get_bool_petscoptions("-blasted_use_symmetric_scaling");
			get_string_petscoptions("-blasted_async_fact_init_type", ctx->factinittype);

			char builditertype[100];
			get_string_petscoptions("-blasted_pc_build_iter_type", builditertype);
			ctx->blditertype = getIterTypeFromString(builditertype);
		}
		else {
			ctx->scale = false;
			strcpy(ctx->factinittype, "NA");
		}
		get_string_petscoptions("-blasted_async_apply_init_type", ctx->applyinittype);
		ctx->threadchunksize = get_int_petscoptions("-blasted_thread_chunk_size");
#ifdef DEBUG
		printf("BLASTed: setupDataFromOptions:\n");
		printf(" fact init type = %s, apply init type = %s", ctx->factinittype, ctx->applyinittype);
		printf(" Fact iter type = %d, apply iter type = %d, ", ctx->blditertype, ctx->aplitertype);
		printf(" Thread chunk size = %d.\n", ctx->threadchunksize); fflush(stdout);
#endif
	}
	else {
		sweeps[0] = 1;
		sweeps[1] = 1;
	}

	ctx->compute_precinfo =
		get_optional_bool_petscoptions("-blasted_compute_preconditioner_info", false);

#ifdef DEBUG
	printf("BLASTed: setupDataFromOptions: Setting up preconditioner with\n");
	printf(" ptype = %d and sweeps = %d,%d.\n", ptype, sweeps[0], sweeps[1]);
#endif

	ctx->bprec = nullptr;
	ctx->prectype = ptype;
	ctx->nbuildsweeps = sweeps[0];
	ctx->napplysweeps = sweeps[1];
	ctx->first_setup_done = true;
	ctx->cputime = ctx->walltime = ctx->factorcputime = ctx->factorwalltime =
		ctx->applycputime = ctx->applywalltime = 0;

	const std::string pcname = std::string("Blasted-") + ctx->prectypestr;

	ierr = PCShellSetName(pc, pcname.c_str()); CHKERRQ(ierr);

	return ierr;
}

/** \brief Generates a BLASTed preconditioner for the preconditioning operator in a sub PC
 *
 * The matrix is assumed to be stored in a sparse (block-)row storage format.
 * \warning We assume that the pc passed here is a subpc, ie, a local preconditioner.
 * \param[in,out] pc PETSc preconditioner context
 */
PetscErrorCode createNewPreconditioner(PC pc)
{
	PetscErrorCode ierr = 0;
	PetscInt firstrow, lastrow, localrows, localcols, globalrows, globalcols;

	// get control structure
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);

	// delete old matrix
	BlastedPreconditioner* precop = reinterpret_cast<BlastedPreconditioner*>(ctx->bprec);
	delete precop;

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

	// ensure diagonal entry locations have been computed; this is necessary for BAIJ matrices
	// as a bonus, check for singular diagonals
	PetscBool diagmissing = PETSC_FALSE;
	PetscInt badrow = -1;
	ierr = MatMissingDiagonal(A, &diagmissing, &badrow); CHKERRQ(ierr);
	if(diagmissing == PETSC_TRUE) {
		SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "! Zero diagonal in (block-)row %d!", badrow);
	}

	const FactoryBase<PetscReal,PetscInt> *const factory
		= (const FactoryBase<PetscReal,PetscInt>*)ctx->bfactory;

	// create appropriate preconditioner and relaxation objects
	AsyncSolverSettings settings;
	settings.prectype = factory->solverTypeFromString(ctx->prectypestr);
	settings.bs = ctx->bs;                         // set in setup_localpreconditioner_blasted below
	settings.blockstorage = ColMajor;              // required for PETSc
	settings.scale = ctx->scale;
	settings.buildtype = ctx->blditertype;
	settings.applytype = ctx->aplitertype;

	// Check if sequential factorization or application was requested and set build/apply sweeps
	setSweeps_checkSeq(ctx, settings);

	settings.thread_chunk_size = ctx->threadchunksize;
	settings.compute_precinfo = ctx->compute_precinfo;
	if(settings.prectype != BLASTED_JACOBI && settings.prectype != BLASTED_LEVEL_SGS
	   && settings.prectype != BLASTED_NO_PREC)
	{
		if(settings.prectype == BLASTED_ILU0 || settings.prectype == BLASTED_SAPILU0 ||
		   settings.prectype == BLASTED_ASYNC_LEVEL_ILU0)
			settings.fact_inittype = getFactInitFromString(ctx->factinittype);
		else
			settings.fact_inittype = INIT_F_NONE;
		settings.apply_inittype = getApplyInitFromString(ctx->applyinittype);
	}

	settings.relax = false;

	// get access to local matrix entries
	const Mat_SeqAIJ *const Adiag = (const Mat_SeqAIJ*)A->data;
	const Mat_SeqBAIJ *const Abdiag = (const Mat_SeqBAIJ*)A->data;

	if(ctx->bs <= 0 || ctx->bs > 5 || ctx->bs == 2)
		SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "BLASTed: Block size %d is not supported!", ctx->bs);

	if(ctx->bs == 1) {
		precop = factory->create_preconditioner(SRMatrixStorage<const PetscReal,const PetscInt>
		                                        (Adiag->i, Adiag->j, Adiag->a, Adiag->diag,
		                                         Adiag->i+1, localrows,
		                                         Adiag->i[localrows], Adiag->i[localrows], 1),
		                                        settings);
	}
	else {
		precop = factory->create_preconditioner(SRMatrixStorage<const PetscReal,const PetscInt>
		                                        (Abdiag->i, Abdiag->j, Abdiag->a, Abdiag->diag,
		                                         Abdiag->i+1, localrows/ctx->bs,
		                                         Abdiag->i[localrows/ctx->bs],
		                                         Abdiag->i[localrows/ctx->bs], ctx->bs),
		                                        settings);
	}


	ctx->bprec = reinterpret_cast<void*>(precop);

	ctx->infolist = NULL;
	if(ctx->compute_precinfo) {
		PrecInfoList *bpinfo = new PrecInfoList;
		bpinfo->infolist.reserve(250);  // some reasonable number of linear solves expected
		ctx->infolist = static_cast<void*>(bpinfo);
	}

	return ierr;
}

/// Re-computes the preconditioner using current values of the associated matrix
PetscErrorCode updatePreconditioner(PC pc)
{
	Blasted_data* ctx;
	int ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);
	BlastedPreconditioner *const precop = reinterpret_cast<BlastedPreconditioner*>(ctx->bprec);
	PrecInfoList *const pilist = static_cast<PrecInfoList*>(ctx->infolist);

	PrecInfo pinfo = precop->compute();

	if(ctx->compute_precinfo)
		pilist->infolist.push_back(pinfo);

	return ierr;
}

extern "C" {

Blasted_data_list newBlastedDataList()
{
	Blasted_data_list b;
	b.ctxlist = NULL;
	b.bfactory = NULL;
	b.size = 0;
	b.factorcputime = b.factorwalltime = b.applycputime = b.applywalltime = 0.0;
	b._defaultfactory = 0;
	return b;
}

void destroyBlastedDataList(Blasted_data_list *const b)
{
	if(b->_defaultfactory == 1) {
		delete (FactoryBase<double,int>*)b->bfactory;
		b->_defaultfactory = 0;
	}

	while(b->ctxlist != NULL) {
		Blasted_data *temp = b->ctxlist;
		b->ctxlist = b->ctxlist->next;

		// Delete the diagnostic info object
		PrecInfoList *list = static_cast<PrecInfoList*>(temp->infolist);
		delete list;
		list = nullptr;

		// delete the Blasted node
		delete temp;

		b->size--;
	}

	if(b->size != 0)
		throw std::logic_error("Could not delete Blasted_data_list properly!");
}

Blasted_data newBlastedDataContext()
{
	Blasted_data ctx;
	ctx.bprec = NULL;
	ctx.infolist = NULL;
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
	BlastedPreconditioner* prec = reinterpret_cast<BlastedPreconditioner*>(ctx->bprec);
	delete prec;

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
		ierr = createNewPreconditioner(pc); CHKERRQ(ierr);
	}

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	ierr = updatePreconditioner(pc); CHKERRQ(ierr);

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	ctx->factorwalltime += (finalwtime - initialwtime);
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	ctx->factorcputime += (finalctime - initialctime);

	return ierr;
}

#if 0
/// Apply any kind of BLASTed operator
PetscErrorCode apply_local_base(Blasted_data *const ctx,
                                const BlastedPreconditioner *const bp, Vec r, Vec z)
{
	PetscErrorCode ierr = 0;
	const PetscReal *ra;
	PetscReal *za;
	PetscInt start, end;
	ierr = VecGetOwnershipRange(r, &start, &end); CHKERRQ(ierr);

#ifdef DEBUG
	if(bp->dim() != end-start) {
		SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
				"BLASTed: Dimension of vector does not match that of the matrix- %d vs %d!\n",
				bp->dim(), end-start);
	}
#endif

	ierr = VecGetArray(z, &za); CHKERRQ(ierr);
	ierr = VecGetArrayRead(r, &ra); CHKERRQ(ierr);

	struct timeval time1, time2;
	gettimeofday(&time1, NULL);
	double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
	double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

	bp->apply(ra,za);

	gettimeofday(&time2, NULL);
	double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
	ctx->applywalltime += (finalwtime - initialwtime);
	double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
	ctx->applycputime += (finalctime - initialctime);

	VecRestoreArrayRead(r, &ra);
	VecRestoreArray(z, &za);

	return ierr;
}
#endif

PetscErrorCode apply_local_blasted(PC pc, Vec r, Vec z)
{
	PetscErrorCode ierr = 0;
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);
	const BlastedPreconditioner *const prec =
		reinterpret_cast<const BlastedPreconditioner*>(ctx->bprec);

	{
		const PetscReal *ra;
		PetscReal *za;
		PetscInt start, end;
		ierr = VecGetOwnershipRange(r, &start, &end); CHKERRQ(ierr);

#ifdef DEBUG
		if(prec->dim() != end-start) {
			SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
			         "BLASTed: Dimension of vector does not match that of the matrix- %d vs %d!\n",
			         prec->dim(), end-start);
		}
#endif

		ierr = VecGetArray(z, &za); CHKERRQ(ierr);
		ierr = VecGetArrayRead(r, &ra); CHKERRQ(ierr);

		struct timeval time1, time2;
		gettimeofday(&time1, NULL);
		double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
		double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

		prec->apply(ra,za);

		gettimeofday(&time2, NULL);
		double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
		ctx->applywalltime += (finalwtime - initialwtime);
		double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
		ctx->applycputime += (finalctime - initialctime);

		VecRestoreArrayRead(r, &ra);
		VecRestoreArray(z, &za);
	}

	return ierr;
}

PetscErrorCode relax_local_blasted(PC pc, Vec rhs, Vec x, Vec w,
                                   const PetscReal rtol, const PetscReal abstol, const PetscReal dtol,
                                   const PetscInt it, const PetscBool guesszero,
                                   PetscInt *const outits, PCRichardsonConvergedReason *const reason)
{
	PetscErrorCode ierr = 0;
	Blasted_data* ctx;
	ierr = PCShellGetContext(pc, (void**)&ctx); CHKERRQ(ierr);

	BlastedPreconditioner *const relaxation =
		reinterpret_cast<BlastedPreconditioner*>(ctx->bprec);

	// set relaxation parameters, though we don't use any of them except the max iterations
	relaxation->setApplyParams({rtol,abstol,dtol,false,it});

	if(guesszero) {
		ierr = VecSet(x, 0.0); CHKERRQ(ierr);
	}

	{
		const PetscReal *ra;
		PetscReal *za;
		PetscInt start, end;
		ierr = VecGetOwnershipRange(rhs, &start, &end); CHKERRQ(ierr);

#ifdef DEBUG
		if(relaxation->dim() != end-start) {
			SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
			         "BLASTed: Dimension of vector does not match that of the matrix- %d vs %d!\n",
			         relaxation->dim(), end-start);
		}
#endif

		ierr = VecGetArray(x, &za); CHKERRQ(ierr);
		ierr = VecGetArrayRead(rhs, &ra); CHKERRQ(ierr);

		struct timeval time1, time2;
		gettimeofday(&time1, NULL);
		double initialwtime = (double)time1.tv_sec + (double)time1.tv_usec * 1.0e-6;
		double initialctime = (double)clock() / (double)CLOCKS_PER_SEC;

		relaxation->apply_relax(ra,za);

		gettimeofday(&time2, NULL);
		double finalwtime = (double)time2.tv_sec + (double)time2.tv_usec * 1.0e-6;
		ctx->applywalltime += (finalwtime - initialwtime);
		double finalctime = (double)clock() / (double)CLOCKS_PER_SEC;
		ctx->applycputime += (finalctime - initialctime);

		VecRestoreArrayRead(rhs, &ra);
		VecRestoreArray(x, &za);
	}

	*reason = PCRICHARDSON_CONVERGED_ITS;
	*outits = it;

	return ierr;
}

int setup_blasted_stack_ext(KSP ksp, const FactoryBase<double,int> *const fctry,
                            Blasted_data_list *const bctv)
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

		ierr = setup_blasted_stack(subksp[0], bctv); CHKERRQ(ierr);
	}
	else if(ismg || isgamg) {
		ierr = KSPSetUp(ksp); CHKERRQ(ierr);
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		PetscInt nlevels;
		ierr = PCMGGetLevels(pc, &nlevels); CHKERRQ(ierr);

		for(int ilvl = 1; ilvl < nlevels; ilvl++) {
			KSP smootherctx;
			ierr = PCMGGetSmoother(pc, ilvl , &smootherctx); CHKERRQ(ierr);
			ierr = setup_blasted_stack(smootherctx, bctv); CHKERRQ(ierr);
			/*
			KSP smootherup, smootherdown;
			ierr = PCMGGetSmootherDown(pc, ilvl , &smootherdown); CHKERRQ(ierr);
			ierr = PCMGGetSmootherUp(pc, ilvl , &smootherup); CHKERRQ(ierr);
			ierr = setup_blasted_stack(smootherdown, bctv, ilvl); CHKERRQ(ierr);
			ierr = setup_blasted_stack(smootherup, bctv, ilvl+nlevels-1); CHKERRQ(ierr);*/
		}
		KSP coarsesolver;
		ierr = PCMGGetCoarseSolve(pc, &coarsesolver); CHKERRQ(ierr);
		ierr = setup_blasted_stack(coarsesolver, bctv); CHKERRQ(ierr);
	}
	else if(isksp) {
		ierr = KSPSetUp(ksp); CHKERRQ(ierr);
		ierr = PCSetUp(pc); CHKERRQ(ierr);
		KSP subksp;
		ierr = PCKSPGetKSP(pc, &subksp); CHKERRQ(ierr);
		ierr = setup_blasted_stack(subksp, bctv); CHKERRQ(ierr);
	}
	else if(isshell) {
		// if the PC is shell, this is the relevant KSP to pass to BLASTed for setup
		std::printf("setup_blasted_stack(): Found valid parent KSP for BLASTed.\n");

		appendBlastedDataContext(bctv, newBlastedDataContext());
		// The new context is appended to the head of the list,
		// so we setup the BLASTed preconditioner using the context at the head of the list

		// set the factory to use - same for each solver context in the tree
		bctv->ctxlist->bfactory = bctv->bfactory;

		ierr = setup_localpreconditioner_blasted(ksp, bctv->ctxlist); CHKERRQ(ierr);
	}

	return ierr;
}

PetscErrorCode setup_blasted_stack(KSP ksp, Blasted_data_list *const bctx)
{
	FactoryBase<double,int> *factory = new SRFactory<double,int>();
	bctx->bfactory = (void*)factory;
	bctx->_defaultfactory = 1;
	return setup_blasted_stack_ext(ksp, factory, bctx);
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

	// set a relaxation application only for supported preconditioners
	if(bctx->prectype != BLASTED_ILU0 &&
	   bctx->prectype != BLASTED_CSC_BGS &&
	   bctx->prectype != BLASTED_NO_PREC)
	{
		ierr = PCShellSetApplyRichardson(subpc, &relax_local_blasted);
		CHKERRQ(ierr);
	}

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
