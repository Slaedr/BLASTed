/** \file csrmatrix.cpp
 * \brief Template instantiations for CSR matrices
 * \author Aditya Kashi
 */

#include "blockmatrices.ipp"

namespace blasted {

template class BSRMatrix<double,int,1>;
//template class BSRMatrix<float,int,1>;
template class CSRMatrixView<double,int>;
//template class CSRMatrixView<float,int>;

}
