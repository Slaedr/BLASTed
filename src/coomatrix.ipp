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

inline
MMDescription getMMDescription(std::ifstream& fin)
{
	std::string line;
	std::getline(fin, line);
	std::vector<std::string> typeofmatrix;
	boost::split(typeofmatrix, line, boost::is_any_of(" "));

	MMDescription descr;

	if(typeofmatrix.size() != 5)
		std::cout << "! Not enough terms in the header!\n";
	if(typeofmatrix[0] != "%%MatrixMarket")
		std::cout << "! Not a matrix market file!\n";

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

template <typename index>
std::vector<index> getSizeFromMatrixMarket(std::ifstream& fin, 
		const MMDescription& descr)
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

template <typename scalar, typename index>
COOMatrix<scalar,index>::COOMatrix()
{ }

template <typename scalar, typename index>
COOMatrix<scalar,index>::~COOMatrix()
{ }

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
		std::cout << "! COOMatrix: readMatrixMarket: Can only read coordinate storage.\n";
	}
	if(descr.scalartype == PATTERN) {
		std::cout << "! COOMatrix: readMatrixMarket: Cannot read pattern matrices.\n";
	}
	if(descr.matrixtype != GENERAL) {
		std::cout << "! COOMatrix: readMatrixMarket: Can only read general matrices.\n";
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
void COOMatrix<scalar,index>::convertToCSR(BSRMatrix<scalar,index,1> *const cmat) const
{ 
	std::vector<index> cinds(nnz);
	for(index i = 0; i < nnz; i++)
		cinds[i] = entries[i].colind;
	
	cmat->setStructure(nrows, cinds.data(), rowptr.data());

	for(index i=0; i < nnz; i++)
	{
		cmat->submitBlock(entries[i].rowind, entries[i].colind, &entries[i].value, 1, 1);
	}
}

template <typename scalar, typename index>
template<int bs, StorageOptions stor>
void COOMatrix<scalar,index>::convertToBSR(RawBSRMatrix<scalar,index> *const bmat) const
{
	static_assert(bs > 0, "Block size must be positive!");
	static_assert(stor == RowMajor || stor == ColMajor, "Invalid storage option!");
	bmat->nbrows = nrows/bs;
	bmat->browptr = new index[bmat->nbrows+1];
	bmat->browptr[0] = -1;
	index bnnz = 0;                             //< Number of nonzero blocks

	for(index irow = 0; irow < nrows; irow++)
	{
		for(index j = rowptr[irow]; j < rowptr[irow+1]; j++)
		{
			//const index colidx = entries[j].colind;
		}
	}
}


