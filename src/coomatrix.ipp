/** \file coomatrix.ipp
 * \brief Implements coordinate matrix formats
 * \author Aditya Kashi
 * 
 * This file is part of BLASTed.
 *   BLASTed is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   BLASTed is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with BLASTed.  If not, see <http://www.gnu.org/licenses/>.
 */

template <typename scalar, typename index>
MMDescription COOMatrix<scalar,index>::getMMDescription(std::ifstream& fin)
{
	std::string line;
	std::getline(std::fin, line);
	std::vector<std::string> typeofmatrix;
	boost::split(typeofmatrix, line, boost::is_any_of(" "));

	MMDescription descr;

	if(typeofmatrix.size() != 5)
		std::cout << "! COOMatrix: Not enough terms in the header!\n";
	if(typeofmatrix[0] != "%%MatrixMarket")
		std::cout << "! COOMatrix: Not a matrix market file!\n";

	if(typeofmatrix[2] == "coordinate") {
		descr.storagetype = COORDINATE;
	}
	else if(typeofmatrix[2] == "array") {
		descr.storagetype = ARRAY;
	}
	else {
		std::cout << "! COOMatrix: Invalid storage type!\n";
		std::abort();
	}

	if(typeofmatrix[3] == "real")
		descr.scalartype = REAL;
	else if(typeofmatrix[3] == "complex")
		descr.scalartype = COMPLEX;
	else if(typeofmatrix[3] == "integer")
		descr.scalartype = INTEGER;
	else if(typeofmatrix[3] == "patter")
		descr.scalartype = PATTERN;
	else {
		std::cout << "! COOMatrix: Invalid scalar type!\n";
		std::abort();
	}

	if(typeofmatrix[4] == "general")
		descr.matrixtype = GENERAL;
	else if(typeofmatrix[4] == "symmetric")
		descr.matrixtype = SYMMETRIC;
	else if(typeofmatrix[4] == "skewsymmetric")
		descr.matrixtype = SKEWSYMMETRIC;
	else if(typeofmatrix[4] == "hermitian")
		descr.matrixtype = HERMITIAN;
	else {
		std::cout << "! COOMatrix: Invalid matrix type!\n";
		std::abort();
	}

	return descr;
}

template <typename scalar, typename index>
void COOMatrix<scalar,index>::readMatrixMarket(const std::string file)
{
	std::ifstream fin(file);
	if(!fin) {
		std::cout << " COOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	MMDescription descr = getMMDescription(fin);

	fin.close();
}

template <typename scalar, typename index>
void COOMatrix<scalar,index>::setStructure(const index n, const index *const vec1, 
		const index *const vec2)
{
}

template <typename scalar, typename index>
void COOMatrix<scalar,index>::apply(const scalar a, const scalar *const x, 
		scalar *const __restrict y) const
{
	// TODO
}

template <typename scalar, typename index>
void COOMatrix<scalar,index>::convertToCSR(BSRMatrix<scalar,index,1> *const cmat) const
{ 
	// TODO
}

template <typename scalar, typename index>
template<int bs>
void COOMatrix<scalar,index>::convertToBSR(BSRMatrix<scalar,index,bs> *const bmat) const
{
	// TODO
}

