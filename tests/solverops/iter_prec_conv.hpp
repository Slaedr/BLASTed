/** \file
 * \brief Common utilities for testing convergence of iteratively-applied preconditioners
 */

#ifndef BLASTED_TESTS_ITER_PREC_CONVERGENCE_UTILS_H
#define BLASTED_TESTS_ITER_PREC_CONVERGENCE_UTILS_H

#include "srmatrixdefs.hpp"
#include "device_container.hpp"
#include "ilu_pattern.hpp"

namespace blasted {

template <int bs>
device_vector<double> getExactILU(const CRawBSRMatrix<double,int> *const mat,
                                  const ILUPositions<int>& plist,
                                  const device_vector<double>& scale);

/// Computes symmetric scaling vector for scalar async ILU
device_vector<double> getScalingVector(const CRawBSRMatrix<double,int> *const mat);

}

#endif
