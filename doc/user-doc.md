Using BLASTed via the PETSc interface
=====================================

There are three main PETSc options controlling the use of BLASTed. These options are used for any preconditioner for which `-pc_type` is set to `shell`, assuming your code is set up to do that (see the next section). The first two options have been created specifically for BLASTed. The third is a standard PETSc option that affects BLASTed usage.

* `-blasted_pc_type` sets the preconditioner to use. Each option below is affected by whether or not the BAIJ matrix type is being used - if so, the block version of the preconditioner is used, otherwise the regular scalar version is used.
  - `jacobi` Jacobi preconditioner or relaxation
  - `gs` Forward Gauss-Seidel iteration
  - `sgs` Symmetric Gauss-Seidel preconditioner or relaxation
  - `ilu0` ILU(0) preconditioner
  - `sapilu0` ILU(0) preconditioner with asynchronous factorization but sequential (forward- or back-substitution) application

* `-blasted_async_sweeps` An integer array specifying the number of asynchronous iterations ("sweeps") to use each time the preconditioner is built and applied. Eg.: `-blasted_async_sweeps 4,3` means the preconditioner is built using 4 asynchronous iterations (sweeps) while it is applied using 3 asynchronous sweeps. If not specified, the default of 1 sweep is used.

* `-blasted_async_fact_init_type` A string specifying what type of initialization to use for asynchronous factorizations. Options: 
  - `init_zero`
  - `init_original`
  - `init_sgs`
* `-blasted_async_apply_init_type` A string specifying what type of initialization to use for asynchronous triangular solves. Options: 
  - `init_zero`
  - `init_jacobi`
* `-blasted_compute_preconditioner info` Boolean value (specifying this option with no value amounts to true; not specifying it at all amounts to false) to request computation and reporting of extra information about the preconditioner to aid analysis. Note that these may be expensive and not well-optimized.

* `-blasted_thread_chunk_size` An integer specifying the number of work-items assigned at a time to a thread in a dynamically-scheduled loop.

* `-mat_type` "aij" (default, if not mentioned) and "baij". If "aij", scalar versions of the algorithms are applied. For example, the preconditioner for Jacobi will be the diagonal of the matrix. If "baij" is specified, point-block versions of the algorithms are carried out. In case of Jacobi, for instance, the preconditioner will be the block-diagonal part of the matrix with the blocks inverted exactly. **NOTE**: this can also affect several other things in your code apart from the behaviour of BLASTed.

In case of algorithms that have both preconditioning and relaxation forms (Jacobi and Gauss-Seidel), which form is applied depends on the PETSc solver structure being used. Specifically, if the local KSP (for which BLASTed is the PC) is KSPRICHARDSON, relaxation is usually applied. The exception is that if either the Richardson damping factor is NOT 1.0, or `-ksp_monitor` is specified, then the preconditioning form is used even with KSPRICHARDSON. For all other local KSPs including PREONLY, only the preconditioning form is used.

Changes needed in your source code
----------------------------------

The BLASTed setup, apply and cleanup functions will have to be set up as explained in the PETSc manual, section 4.4.7 on shell preconditioners. Keep in mind that since BLASTed preconditioners are meant to be local, they should be set as sub-preconditioners (`-sub_pc_type`) to a global preconditioner such as subdomain-block Jacobi (`-pc_type bjacobi`) or additive Schwarz (`-pc_type asm`). This means that in your code, the shell preconditioner to be set up must be obtained from the sub KSPs of the global preconditioner. In addition to the functions for setup, application and destruction, the context has to be supplied to `PCSHELL`. For BLASTed, this is the `Blasted_data_list` type defined in `include/blasted_petsc.h`. An object of this type must be created by the user.

All this can be mostly automated using the `setup_blasted_stack` function in `include/blasted_petsc.h`. Just create a `Blasted_data_list` object and pass it to this function along with the outer-most (global) `KSP` under which to set BLASTed as the subdomain solver, local multigrid smoother or any other local solver component. If you use this function, the block size will be taken from the preconditioning operator associated with the `KSP` argument. Thus, `KSPSetOperators` must be called and the block size must be set (either using `MatSetBlockSize` or `-mat_block_size`) beforehand.
