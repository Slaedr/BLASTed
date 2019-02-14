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

#include <cassert>
#include <algorithm>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/align/aligned_alloc.hpp>
#include <coomatrix.hpp>

namespace blasted {

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

/// Returns a [description](\ref MMDescription) of the matrix if it's in Matrix Market format
inline
MMDescription getMMDescription(std::ifstream& fin)
{
	std::string line;
	std::getline(fin, line);
	std::vector<std::string> typeofmatrix;
	boost::split(typeofmatrix, line, boost::is_any_of(" "));

	MMDescription descr;

	if(typeofmatrix.size() != 5)
		throw MatrixReadException("! Not enough terms in the header!");
	if(typeofmatrix[0] != "%%MatrixMarket")
		throw MatrixReadException("! Not a matrix market file!");

	if(typeofmatrix[2] == "coordinate") {
		descr.storagetype = COORDINATE;
	}
	else if(typeofmatrix[2] == "array") {
		descr.storagetype = ARRAY;
	}
	else {
		std::cout << "! Invalid storage type!\n";
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
		std::cout << "! Invalid scalar type!\n";
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
		std::cout << "! Invalid matrix type!\n";
		std::abort();
	}

	return descr;
}

/// Returns a vector containing size information of a matrix in a Matrix Market file
template <typename index>
std::vector<index> getSizeFromMatrixMarket(std::ifstream& fin, const MMDescription& descr)
{
	// read and discard lines until the first line that does not begin with '%'
	std::string line = "%";
	while(line[0] == '%')
		std::getline(fin, line);

	// parse matrix size line
	std::vector<std::string> sizestr;
	boost::split(sizestr, line, boost::is_any_of(" "));

	if(descr.storagetype == COORDINATE) {
		if(sizestr.size() < 3) {
			std::cout << "! getSizeFromMatrixMarket: Not enough size information!\n";
			std::abort();
		}
	}
	else {
		if(sizestr.size() < 2) {
			std::cout << "! getSizeFromMatrixMarket: Not enough size information!\n";
			std::abort();
		}
	}

	std::vector<index> sizes(sizestr.size());
	for(int i = 0; i < static_cast<int>(sizes.size()); i++)
	{
		try {
			sizes[i] = std::stoi(sizestr[i]);
		}
		catch(const std::invalid_argument& e) {
			std::cout << "! getSizeFromMatrixMarket: Invalid size!!\n";
			std::abort();
		}
		catch(const std::out_of_range& e) {
			std::cout << "! getSizeFromMatrixMarket: Size is too large for the type of index!!\n";
			std::abort();
		}
	}

	return sizes;
}

template <typename scalar>
std::vector<scalar> readDenseMatrixMarket(const std::string file)
{
	std::ifstream fin(file);
	if(!fin) {
		std::cout << " readDenseMatrixMarket: File could not be opened to read!\n";
		std::abort();
	}

	const MMDescription descr = getMMDescription(fin);
	if(descr.matrixtype != GENERAL) {
		fin.close();
		throw "Matrix should be general!";
	}
	if(descr.storagetype != ARRAY) {
		fin.close();
		throw "Matrix should be stored as dense!";
	}

	const std::vector<int> sizes = getSizeFromMatrixMarket<int>(fin,descr);
	if(sizes.size() < 2) {
		fin.close();
		throw "Size vector has less than 2 entries!";
	}

	std::vector<scalar> vals(sizes[0]*sizes[1]);
	for(int i = 0; i < sizes[0]*sizes[1]; i++)
		fin >> vals[i];

	fin.close();
	return vals;
}

template std::vector<double> readDenseMatrixMarket<double>(const std::string file);
template std::vector<float> readDenseMatrixMarket<float>(const std::string file);

template <typename scalar, typename index>
COOMatrix<scalar,index>::COOMatrix()
{ }

template <typename scalar, typename index>
COOMatrix<scalar,index>::~COOMatrix()
{ }

template <typename scalar, typename index>
index COOMatrix<scalar,index>::numrows() const {return nrows; }

template <typename scalar, typename index>
index COOMatrix<scalar,index>::numcols() const {return ncols; }

template <typename scalar, typename index>
index COOMatrix<scalar,index>::numnonzeros() const {return nnz; }

/** Currently reads only general mtx matrices, not symmetric or skew-symmetric.
 */
template <typename scalar, typename index>
void COOMatrix<scalar,index>::readMatrixMarket(const std::string file)
{
	std::ifstream fin(file);
	if(!fin) {
		std::cout << " COOMatrix: File could not be opened to read!\n";
		std::abort();
	}

	const MMDescription descr = getMMDescription(fin);
	if(descr.storagetype != COORDINATE) {
		throw MatrixReadException("! COOMatrix: readMatrixMarket: Can only read coordinate storage.");
	}
	if(descr.scalartype == PATTERN) {
		throw MatrixReadException("! COOMatrix: readMatrixMarket: Cannot read pattern matrices.");
	}
	if(descr.matrixtype != GENERAL) {
		throw MatrixReadException("! COOMatrix: readMatrixMarket: Can only read general matrices.");
	}

	std::vector<index> sizes = getSizeFromMatrixMarket<index>(fin,descr);

	nnz = sizes[2];
	nrows = sizes[0];
	ncols = sizes[1];
	entries.resize(nnz);

	// read the entries
	for(index i = 0; i < nnz; i++) {
		index ri, ci;
		fin >> ri >> ci >> entries[i].value;
 		entries[i].rowind = ri-1;
		entries[i].colind = ci-1;
	}

	fin.close();

	// sort by row
	std::sort(entries.begin(), entries.end(), 
			[](Entry<scalar,index> a, Entry<scalar,index> b) { return a.rowind < b.rowind; } );

	// get row pointers- note that we assume there's at least one element in each row
	rowptr.resize(nrows+1);
	std::vector <typename std::vector<Entry<scalar,index>>::iterator> rowits(nrows+1);
	rowptr[0] = 0;
	rowits[0] = entries.begin();
	
	index k = 1;                 //< Keeps track of current entry
	index nextrow = 1;           //< Keeps track of the number of rows detected
	for(auto it = entries.begin()+1; it != entries.end(); ++it)
	{
		if( (*it).rowind != (*(it-1)).rowind ) {
			rowits[nextrow] = it;
			rowptr[nextrow] = k;
			nextrow++;
		}

		k++;
	}
	
	assert(nextrow == nrows);       //< The previous row should have been the last, ie. nrows-1
	rowits[nrows] = entries.end();
	rowptr[nrows] = nnz;

	// Sort each row by columns
	for(index i = 0; i < nrows; i++)
	{
		std::sort(rowits[i],rowits[i+1], 
				[](Entry<scalar,index> a, Entry<scalar,index> b) { return a.colind < b.colind; } );
	}
}

template <typename scalar, typename index>
void COOMatrix<scalar,index>::convertToCSR(RawBSRMatrix<scalar,index> *const cmat) const
{ 
	cmat->nbrows = nrows;
	cmat->browptr = (index*)aligned_alloc(CACHE_LINE_LEN,(nrows+1)*sizeof(index));
	cmat->bcolind = (index*)aligned_alloc(CACHE_LINE_LEN,nnz*sizeof(index));
	cmat->vals = (scalar*)aligned_alloc(CACHE_LINE_LEN,nnz*sizeof(scalar));
	cmat->diagind = (index*)aligned_alloc(CACHE_LINE_LEN,nrows*sizeof(index));

	for(index i = 0; i < nnz; i++) {
		cmat->bcolind[i] = entries[i].colind;
		cmat->vals[i] = entries[i].value;
	}
	for(index i = 0; i < nrows+1; i++)
		cmat->browptr[i] = rowptr[i];

	for(index i=0; i < nrows; i++)
	{
		for(index j = rowptr[i]; j < rowptr[i+1]; j++)
		{
			if(entries[j].colind == entries[j].rowind)
				cmat->diagind[i] = j;
		}
	}
}

template <typename scalar, typename index>
template<int bs, StorageOptions stor>
void COOMatrix<scalar,index>::convertToBSR(RawBSRMatrix<scalar,index> *const bmat) const
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stor == RowMajor || stor == ColMajor, "Invalid storage option!");

	// only for square matrices
	assert(nrows==ncols);

	// the dimension of the matrix must be a multiple of the block size
	assert(nrows % bs == 0);

	bmat->nbrows = nrows/bs;
	bmat->browptr = (index*)aligned_alloc(CACHE_LINE_LEN,(bmat->nbrows+1)*sizeof(index));
	bmat->diagind = (index*)aligned_alloc(CACHE_LINE_LEN,bmat->nbrows*sizeof(index));
	for(index i = 0; i < bmat->nbrows; i++)
		bmat->diagind[i] = -1;
	for(index i = 0; i < bmat->nbrows+1; i++)
		bmat->browptr[i] = 0;
	index bnnz = 0;                             //< Running count of number of nonzero blocks

	std::vector<index> bcolidxs;
	bcolidxs.reserve(nnz/bs);
	std::vector<bool> tallybrows(bmat->nbrows, false);

	/** If nonzeros in each row were sorted by column initially, 
	 * blocks in each block-row would end up sorted by block-column.
	 */

	for(index irow = 0; irow < nrows; irow++)
	{
		const index curbrow = irow/bs;
		for(index j = rowptr[irow]; j < rowptr[irow+1]; j++)
		{
			const index curcol = entries[j].colind;
			const index curbcol = curcol/bs;

			if(!tallybrows[curbrow]) {
				bmat->browptr[curbrow] = bnnz;
				tallybrows[curbrow] = true;
			}

			// find the current block-column of the current block-row in the array of column indices
			auto it = std::find(bcolidxs.begin()+bmat->browptr[curbrow], bcolidxs.end(), curbcol);
			
			// if it does not exist, add the current block col index
			if(it == bcolidxs.end()) {
				bcolidxs.push_back(curbcol);
				if(curbcol == curbrow)
					bmat->diagind[curbrow] = bnnz;
				bnnz++;
			}
		}
	}

	//std::cout << "convertToBSR: Number of nonzero blocks = " << bnnz << std::endl;
	bmat->browptr[bmat->nbrows] = bnnz;

	// fix browptr for empty block rows
	for(index i = bmat->nbrows-1; i > 0; i--)
		if(bmat->browptr[i] == 0)
			bmat->browptr[i] = bmat->browptr[i+1];

	bmat->bcolind = (index*)aligned_alloc(CACHE_LINE_LEN, bnnz*sizeof(index));
	bmat->vals = (scalar*)aligned_alloc(CACHE_LINE_LEN, bnnz*bs*bs*sizeof(scalar));

	for(index i = 0; i < bnnz*bs*bs; i++)
		bmat->vals[i] = 0;

	for(index i = 0; i < bnnz; i++)
		bmat->bcolind[i] = bcolidxs[i];

	// copy non-zero values
	for(index irow = 0; irow < nrows; irow++)
	{
		const index curbrow = irow/bs;
		for(index j = rowptr[irow]; j < rowptr[irow+1]; j++)
		{
			const index curcol = entries[j].colind;
			const index curbcol = curcol/bs;
			const index offset = stor==RowMajor ? 
				(irow-curbrow*bs)*bs + curcol-curbcol*bs : (curcol-curbcol*bs)*bs + irow-curbrow*bs;
			
			index *const bcptr = std::find(
					bmat->bcolind + bmat->browptr[curbrow], 
					bmat->bcolind + bmat->browptr[curbrow+1], 
					curbcol);

			if(bcptr == bmat->bcolind + bmat->browptr[curbrow+1]) {
				std::cout << "! convertToBSR: Error: Memory not found for " << irow << ", " 
					<< curcol << std::endl;
				std::abort();
			}

			*(bmat->vals + (ptrdiff_t)(bcptr-bmat->bcolind)*bs*bs + offset) = entries[j].value;
		}
	}
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs> constructBSRMatrixFromMatrixMarketFile(const std::string file)
{
	COOMatrix<scalar,index> coomat;
	coomat.readMatrixMarket(file);
	RawBSRMatrix<scalar,index> rmat;
	if(bs == 1)
		coomat.convertToCSR(&rmat);
	else
		coomat.template convertToBSR<bs,RowMajor>(&rmat);
	BSRMatrix<scalar,index,bs> bmat(rmat);
	return bmat;
}

MatrixReadException::MatrixReadException(const std::string& msg) : std::runtime_error(msg)
{ }

template class COOMatrix<double,int>;
template class COOMatrix<float,int>;

template
void COOMatrix<double,int>::convertToBSR<2,RowMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<3,RowMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<4,RowMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<5,RowMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<6,RowMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<7,RowMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<2,ColMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<3,ColMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<4,ColMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<5,ColMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<6,ColMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<double,int>::convertToBSR<7,ColMajor>(RawBSRMatrix<double,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<2,RowMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<3,RowMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<4,RowMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<5,RowMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<6,RowMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<7,RowMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<2,ColMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<3,ColMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<4,ColMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<5,ColMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<6,ColMajor>(RawBSRMatrix<float,int> *const bmat) const;
template
void COOMatrix<float,int>::convertToBSR<7,ColMajor>(RawBSRMatrix<float,int> *const bmat) const;

template 
BSRMatrix<double,int,1> constructBSRMatrixFromMatrixMarketFile(const std::string file);
template 
BSRMatrix<double,int,7> constructBSRMatrixFromMatrixMarketFile(const std::string file);

#ifdef BUILD_BLOCK_SIZE
template 
BSRMatrix<double,int,BUILD_BLOCK_SIZE>
constructBSRMatrixFromMatrixMarketFile(const std::string file);
#endif

}
