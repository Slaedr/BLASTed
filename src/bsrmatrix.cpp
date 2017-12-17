/** \file bsrmatrix.cpp
 * \brief Template instantiations for BSR matrices
 * \author Aditya Kashi
 */

#include "blockmatrices.ipp"

namespace blasted {

//template class BSRMatrix<double,int,2>;
template class BSRMatrix<double,int,3>;
//template class BSRMatrix<double,int,4>;
//template class BSRMatrix<double,int,5>;
//template class BSRMatrix<double,int,6>;
template class BSRMatrix<double,int,7>;
/*template class BSRMatrix<float,int,2>;
template class BSRMatrix<float,int,3>;
template class BSRMatrix<float,int,4>;
template class BSRMatrix<float,int,5>;
template class BSRMatrix<float,int,6>;
template class BSRMatrix<float,int,7>;*/

//template class BSRMatrixView<double,int,2,RowMajor>;
template class BSRMatrixView<double,int,3,RowMajor>;
template class BSRMatrixView<double,int,4,RowMajor>;
//template class BSRMatrixView<double,int,5,RowMajor>;
//template class BSRMatrixView<double,int,6,RowMajor>;
template class BSRMatrixView<double,int,7,RowMajor>;
/*template class BSRMatrixView<float,int,2,RowMajor>;
template class BSRMatrixView<float,int,3,RowMajor>;
template class BSRMatrixView<float,int,4,RowMajor>;
template class BSRMatrixView<float,int,5,RowMajor>;
template class BSRMatrixView<float,int,6,RowMajor>;
template class BSRMatrixView<float,int,7,RowMajor>;*/

// Col major

//template class BSRMatrixView<double,int,2,ColMajor>;
template class BSRMatrixView<double,int,3,ColMajor>;
template class BSRMatrixView<double,int,4,ColMajor>;
template class BSRMatrixView<double,int,5,ColMajor>;
//template class BSRMatrixView<double,int,6,ColMajor>;
template class BSRMatrixView<double,int,7,ColMajor>;

/*template class BSRMatrixView<float,int,2,RowMajor>;
template class BSRMatrixView<float,int,3,RowMajor>;
template class BSRMatrixView<float,int,4,RowMajor>;
template class BSRMatrixView<float,int,5,RowMajor>;
template class BSRMatrixView<float,int,6,RowMajor>;
template class BSRMatrixView<float,int,7,RowMajor>;
template class BSRMatrixView<float,int,2,ColMajor>;
template class BSRMatrixView<float,int,3,ColMajor>;
template class BSRMatrixView<float,int,4,ColMajor>;
template class BSRMatrixView<float,int,5,ColMajor>;
template class BSRMatrixView<float,int,6,ColMajor>;
template class BSRMatrixView<float,int,7,ColMajor>;*/

}
