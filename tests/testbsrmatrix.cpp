/** \file testbsrmatrix.cpp
 * \brief Implementation of tests for block matrix operations
 * \author Aditya Kashi
 */

#undef NDEBUG

#include "testbsrmatrix.hpp"

template <int bs>
TestBSRMatrix<bs>::TestBSRMatrix(const int nb, const int na)
	: BSRMatrix<double,int,bs>(nb,na)
{ }
