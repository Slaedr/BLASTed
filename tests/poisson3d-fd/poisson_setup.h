/** \file
 * \brief Header for getting LHS and RHS of discrete Poisson problem
 * \author Aditya Kashi
 */

#ifndef BLASTED_TEST_POISSON_SETUP_H
#define BLASTED_TEST_POISSON_SETUP_H

#include "../testutils.h"
#include <petscmat.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Returns a discrete Poisson system given a Poisson problem configuration file
/** Assumes PETSc is initialized.
 */
DiscreteLinearProblem setup_poisson_problem(const char *const confile);

#ifdef __cplusplus
}
#endif

#endif
