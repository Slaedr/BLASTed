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

/// Depending on the first argument, either sets up linear problem from binary files or from Poisson config
/** The first argument must be either "poisson" or "file".
 * \param[in] argstart The index into the argv array from which to start reading arguments.
 */
DiscreteLinearProblem generateDiscreteProblem(const int argc, char *argv[], const int argstart);

#ifdef __cplusplus
}
#endif

#endif
