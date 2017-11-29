/** \file testsolve.hpp
 * \brief Tests for preconditioning operations
 * \author Aditya Kashi
 * \date 2017-11-29
 */

#ifndef TESTSOLVE_H
#define TESTSOLVE_H

template<int bs>
int testSolveRichardson(const std::string precontype,
		const std::string mattype, const std::string storageorder, 
		const std::string matfile, const std::string xfile, const std::string bfile);

#endif
