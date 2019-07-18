/** \file testcsrmatrix.cpp
 * \brief Implementation of tests for CSR matrix operations
 * \author Aditya Kashi
 */

#undef NDEBUG

#include <vector>
#include <fstream>
#include <float.h>
#include <coomatrix.hpp>
#include "testcsrmatrix.hpp"

int testCSRMatMult(const std::string type,
		const std::string matfile, const std::string xvec, const std::string prodvec)
{
	RawBSRMatrix<double,int> rm;
	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);
	coom.convertToCSR(&rm);

	const std::vector<double> x = readDenseMatrixMarket<double>(xvec);
	const std::vector<double> ans = readDenseMatrixMarket<double>(prodvec);
	std::vector<double> y(rm.nbrows);
	
	AbstractLinearOperator<double,int>* testmat = nullptr;
	if(type == "view") {
		testmat = new CSRMatrixView<double,int>(rm.nbrows,
		                                        rm.browptr,rm.bcolind,rm.vals,rm.diagind);
	}
	else
		testmat = new BSRMatrix<double,int,1>(rm.nbrows,rm.browptr,rm.bcolind,rm.vals,rm.diagind);

	testmat->apply(x.data(), y.data());

	for(int i = 0; i < rm.nbrows; i++) {
		assert(std::fabs(y[i]-ans[i]) < 10*DBL_EPSILON);
	}

	delete testmat;
	alignedDestroyRawBSRMatrix(rm);

	return 0;
}

