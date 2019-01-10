/** \file
 * \brief C++ interface to BLASTEd's PETSc interface for extensibility
 * \author Aditya Kashi
 */

#ifndef BLASTED_PETSC_EXT_H
#define BLASTED_PETSC_EXT_H

#include "blasted_petsc.h"
#include "solverfactory.hpp"

namespace blasted {

/// Recursive function to set the BLASTed preconditioner wherever possible in the entire solver stack
/** Finds shell PCs and sets BLASTed as the preconditioner for each of them.
 * Assumptions:
 *  - If multigrid occurs, the same smoother (and the same number of sweeps) is used for both
 *   pre- and post-smoothing.
 *
 * \param ksp A PETSc solver context
 * \param factory The factory object to be used creating the preconditioner for each relevant slot
 *   in the solver tree of the given KSP.
 * \param bctx The BLASTed structures that store required settings and data; must be created by
 * \ref newBlastedDataList. It should later be deleted by the user, calling \ref destroyBlastedDataList
 *   after the ksp has been destroyed.
 */
int setup_blasted_stack_ext(KSP ksp, const FactoryBase<double,int>& factory,
                            Blasted_data_list *const bctx);

}

#endif
