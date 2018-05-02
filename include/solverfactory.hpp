/** \file solverfactory.hpp
 * \brief Functions to return a preconditioner or relaxation object from runtime parameters
 * \author Aditya Kashi
 * \date 2018-04
 */

#ifndef BLASTED_SOLVERFACTORY_H
#define BLASTED_SOLVERFACTORY_H

#include <string>
#include <map>

#include "solverops_base.hpp"

namespace blasted {

/** \defgroup prectypelabels Strings for selecting the basic preconditioning method
 * @{
 */
/// Identity
const std::string noprecstr = "none";
/// Jacobi
const std::string jacobistr = "jacobi";
/// Gauss-Seidel
const std::string gsstr = "gs";
/// SGS
const std::string sgsstr = "sgs";
/// ILU(0)
const std::string ilu0str = "ilu0";
/// Sequentially applied ILU(0)
const std::string sapilu0str = "sapilu0";
/** @} */

/** \defgroup blockorderlabels Strings for selecting the storage order within dense blocks
 * @{
 */
const std::string rowmajorstr = "rowmajor";
const std::string colmajorstr = "colmajor";
/** @} */

/** \defgroup integerkeys Strings which are keys in an integer parameter list
 * @{
 */
const std::string ndimstr = "ndim";
const std::string nbuildsweeps = "nbuildsweeps";
const std::string napplysweeps = "napplysweeps";
/** @} */

/// Creates a preconditioner or relaxation object, for matrices stored by sparse rows
/** \param precstr A string describing what preconditioner is needed \sa prectypelabels
 * \param bs Block size to create a dense block preconditioner
 * \param blockstorage String decsribing the layout within a dense block \sa blockorderlabels
 * \param relaxation True if the relaxation form of the algorithm is required.
 * \param intParamList A parameter list of integer parameters. The keys needed depends on the type of
 *  preconditioner requested in \ref precstr. \sa integerkeys
 * \param floatParamList A parameter list of integer parameters. What keys are needed depends on 
 *  the type of preconditioner requested in \ref precstr
 *
 * Throws an instance of invalid_argument in case of invalid argument(s). Throws out_of_range if a
 * required key does not exist in the parameter lists.
 */
template <typename scalar, typename index>
SRPreconditioner<scalar,index> *create_sr_preconditioner
	(const std::string precstr, const int bs, const std::string blockstorage,
	 const bool relaxation,
	 const std::map<std::string,int>& intParamList,
	 const std::map<std::string,double>& floatParamList);

}

#endif
