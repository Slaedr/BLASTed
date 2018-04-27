/** \file solverops_jacobi.cpp
 * \brief Template instantiations for (block-) Jacobi operations
 * \author Aditya Kashi
 */

#include "solverops_jacobi.ipp"

namespace blasted {

	template class JacobiSRPreconditioner<double,int,1,RowMajor>;

	template class JacobiSRPreconditioner<double,int,3,ColMajor>;
	template class JacobiSRPreconditioner<double,int,4,ColMajor>;
	template class JacobiSRPreconditioner<double,int,5,ColMajor>;
	template class JacobiSRPreconditioner<double,int,7,ColMajor>;

	template class JacobiSRPreconditioner<double,int,3,RowMajor>;
	template class JacobiSRPreconditioner<double,int,4,RowMajor>;
	template class JacobiSRPreconditioner<double,int,7,RowMajor>;
}
