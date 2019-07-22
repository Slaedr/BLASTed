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
	COOMatrix<double,int> coom;
	coom.readMatrixMarket(matfile);

	const std::vector<double> x = readDenseMatrixMarket<double>(xvec);
	const std::vector<double> ans = readDenseMatrixMarket<double>(prodvec);
	
	AbstractLinearOperator<double,int>* testmat = nullptr;
	if(type == "view") {
		testmat = new CSRMatrixView<double,int>
			(move_to_const<double,int>(getSRMatrixFromCOO<double,int,1>(coom, "")));
	}
	else
		testmat = new BSRMatrix<double,int,1>(getSRMatrixFromCOO<double,int,1>(coom, ""));

	std::vector<double> y(testmat->dim());
	testmat->apply(x.data(), y.data());

	for(int i = 0; i < testmat->dim(); i++) {
		assert(std::fabs(y[i]-ans[i]) < 10*DBL_EPSILON);
	}

	delete testmat;

	return 0;
}

