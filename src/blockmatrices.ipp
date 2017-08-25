/** \file blockmatrices.ipp
 * \brief Implementation of block matrix methods,
 * including the specialized set of methods for block size 1.
 * \author Aditya Kashi
 * \date 2017-08
 */

template <typename scalar, typename index, int bs>
BSRMatrix<scalar,index,bs>::BSRMatrix(const index n_brows,
		const index *const bcinds, const index *const brptrs,
		const short n_buildsweeps, const short n_applysweeps)
	: LinearOperator<scalar,index>('b'), vals(nullptr), isAllocVals(true), nbrows{n_brows},
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{500}
{
	// compile-time filter for unsigned index types
	index check_var{-1};
	if(check_var != -1) std::cout << "! BSRMatrix: Invalid index type!\n";

	constexpr int bs2 = bs*bs;
	browptr = new index[nbrows+1];
	bcolind = new index[brptrs[nbrows]];
	diagind = new index[nbrows];
	vals = new scalar[brptrs[nbrows]*bs2];
	for(index i = 0; i < nbrows+1; i++)
		browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[nbrows]; i++)
		bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < nbrows; irow++) 
	{
		for(index j = browptr[irow]; j < browptr[irow+1]; j++)
			if(bcolind[j] == irow) {
				diagind[irow] = j;
				break;
			}
	}
	std::cout << "BSRMatrix: Setup with matrix with " << nbrows << " block-rows,\n    "
		<< nbuildsweeps << " build- and " << napplysweeps << " apply- async sweep(s)\n";
}

template <typename scalar, typename index, int bs>
BSRMatrix<scalar, index, bs>::~BSRMatrix()
{
	if(isAllocVals)
		delete [] vals;
	if(bcolind)
		delete [] bcolind;
	if(browptr)
		delete [] browptr;
	if(diagind)
		delete [] diagind;
	bcolind = browptr = diagind = nullptr;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setAllZero()
{
	const index nnz = browptr[nbrows]*bs*bs;
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < nnz; i++)
		vals[i] = 0;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::setDiagZero()
{
	constexpr int bs2 = bs*bs;

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
#pragma omp simd
		for(index jj = diagind[irow]*bs2; jj < (diagind[irow]+1)*bs2; jj++)
			vals[jj] = 0;
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::submitBlock(const index starti, const index startj,
		const scalar *const buffer, const index param1, const index param2) 
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;
	for(index j = browptr[startr]; j < browptr[startr+1]; j++) {
		if(bcolind[j] == startc) 
		{
			for(int k = 0; k < bs2; k++)
				vals[j*bs2 + k] = buffer[k];
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateDiagBlock(const index starti,
		const scalar *const buffer, const index param)
{
	constexpr int bs2 = bs*bs;
	const index startr = starti/bs;
	const index pos = diagind[startr];
	for(int k = 0; k < bs2; k++)
#pragma omp atomic update
		vals[pos*bs2 + k] += buffer[k];
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::updateBlock(const index starti, const index startj,
		const scalar *const buffer, const index param1, const index param2)
{
	bool found = false;
	const index startr = starti/bs, startc = startj/bs;
	constexpr int bs2 = bs*bs;

	for(index j = browptr[startr]; j < browptr[startr+1]; j++) {
		if(bcolind[j] == startc) {
			for(int k = 0; k < bs2; k++)
			{
#pragma omp atomic update
				vals[j*bs2 + k] += buffer[k];
			}
			found = true;
		}
	}
	if(!found)
		std::cout << "! BSRMatrix: submitBlock: Block not found!!\n";
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::apply(const scalar a, const scalar *const xx,
                                       scalar *const __restrict yy) const
{
	Eigen::Map<const Vector<scalar>> x(xx, nbrows*bs);
	Eigen::Map<Vector<scalar>> y(yy, nbrows*bs);
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, browptr[nbrows]*bs, bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		y.SEG<bs>(irow*bs) = Vector<scalar>::Zero(bs);

		// loop over non-zero blocks of this block-row
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			// multiply the blocks with corresponding sub-vectors
			const index jcol = bcolind[jj];
			y.SEG<bs>(irow*bs).noalias() 
				+= a * data.BLK<bs,bs>(jj*bs,0) * x.SEG<bs>(jcol*bs);
			
			/*for(int i = 0; i < bs; i++)
				for(int j = 0; j < bs; j++)
					y[irow*bs+i] += a * data[jj*bs2 + i*bs+j] * x[jcol*bs+j];*/
		}
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::gemv3(const scalar a, const scalar *const __restrict__ xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
	Eigen::Map<const Vector<scalar>> x(xx, nbrows*bs);
	Eigen::Map<const Vector<scalar>> y(yy, nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, nbrows*bs);
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, browptr[nbrows]*bs, bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		z.SEG<bs>(irow*bs) = b * y.SEG<bs>(irow*bs);

		// loop over non-zero blocks of this block-row
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			const index jcol = bcolind[jj];
			z.SEG<bs>(irow*bs).noalias() += 
				a * data.BLK<bs,bs>(jj*bs,0) * x.SEG<bs>(jcol*bs);
		}
	}
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precJacobiSetup()
{
	if(dblocks.size() <= 0) {
		dblocks.resize(nbrows*bs,bs);
#if DEBUG==1
		std::cout << " BSRMatrix: precJacobiSetup(): Allocating.\n";
#endif
	}
	
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, browptr[nbrows]*bs, bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		dblocks.BLK<bs,bs>(irow*bs,0) = data.BLK<bs,bs>(diagind[irow]*bs,0).inverse();
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precJacobiApply(const scalar *const rr, 
                                                 scalar *const __restrict zz) const
{
	Eigen::Map<const Vector<scalar>> r(rr, nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, nbrows*bs);
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, browptr[nbrows]*bs, bs);

#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		z.SEG<bs>(irow*bs).noalias() = dblocks.BLK<bs,bs>(irow*bs,0) * r.SEG<bs>(irow*bs);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::allocTempVector()
{
	ytemp.resize(nbrows*bs,1);
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precSGSApply(const scalar *const rr, 
                                              scalar *const __restrict zz) const
{
	Eigen::Map<const Vector<scalar>> r(rr, nbrows*bs);
	Eigen::Map<Vector<scalar>> z(zz, nbrows*bs);
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, browptr[nbrows]*bs, bs);

	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < nbrows; irow++)
		{
			Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();

			for(index jj = browptr[irow]; jj < diagind[irow]; jj++)
				inter += data.BLK<bs,bs>(jj*bs,0)*ytemp.SEG<bs>(bcolind[jj]*bs);

			ytemp.SEG<bs>(irow*bs) = dblocks.BLK<bs,bs>(irow*bs,0) 
			                                          * (r.SEG<bs>(irow*bs) - inter);
		}
	}

	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = nbrows-1; irow >= 0; irow--)
		{
			Matrix<scalar,bs,1> inter = Matrix<scalar,bs,1>::Zero();
			
			// compute U z
			for(index jj = diagind[irow]+1; jj < browptr[irow+1]; jj++)
				inter += data.BLK<bs,bs>(jj*bs,0) * z.SEG<bs>(bcolind[jj]*bs);

			// compute z = D^(-1) (D y - U z) for the irow-th block-segment of z
			z.SEG<bs>(irow*bs) = dblocks.BLK<bs,bs>(irow*bs,0) 
				* ( data.BLK<bs,bs>(diagind[irow]*bs,0)*ytemp.SEG<bs>(irow*bs) - inter );
		}
	}
}

/// Search through inner indices
/** Finds the position in the index arary that has value indtofind
 * Searches between positions
 * \param[in] start, and
 * \param[in] end
 */
template <typename index>
static inline void inner_search(const index *const aind, 
		const index start, const index end, 
		const index indtofind, index *const pos)
{
	for(index j = start; j < end; j++) {
		if(aind[j] == indtofind) {
			*pos = j;
			break;
		}
	}
}

/** There is currently no pre-scaling of the original matrix A, unlike the point ILU0.
 * It will probably be too expensive to carry out a row-column scaling like in the point case.
 * However, we could try a row scaling.
 */
template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precILUSetup()
{
	Eigen::Map<const Matrix<scalar,Dynamic,bs,RowMajor>> data(vals, browptr[nbrows]*bs, bs);

	if(iluvals.size() < browptr[nbrows]*bs*bs)
	{
		std::printf(" BSRMatrix: precILUSetup(): First-time setup\n");

		// Allocate lu
		iluvals.resize(browptr[nbrows]*bs,bs);
#pragma omp parallel for simd default(shared)
		for(int j = 0; j < browptr[nbrows]*bs*bs; j++) {
			iluvals.data()[j] = vals[j];
		}

		// intermediate array for the solve part; NOT ZEROED
		if(ytemp.size() < nbrows*bs)
			ytemp.resize(nbrows*bs);
		else
			std::cout << "! BSRMatrix: precILUSetup(): Temp vector is already allocated!\n";
	}

	// compute L and U
	/** Note that in the factorization loop, the variable pos is initially set negative.
	 * If index is an unsigned type, that might be a problem. However,
	 * it should usually be okay as we are only comparing equality later.
	 */
	
	for(short isweep = 0; isweep < nbuildsweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < nbrows; irow++)
		{
			for(index j = browptr[irow]; j < browptr[irow+1]; j++)
			{
				if(irow > bcolind[j])
				{
					Matrix<scalar,bs,bs> sum = data.BLK<bs,bs>(j*bs,0);

					for(index k = browptr[irow]; 
					    (k < browptr[irow+1]) && (bcolind[k] < bcolind[j]); 
					    k++  ) 
					{
						index pos = -1;
						inner_search<index> ( bcolind, 
							diagind[bcolind[k]], browptr[bcolind[k]+1], bcolind[j], &pos );

						if(pos == -1) continue;

						sum.noalias() -= iluvals.BLK<bs,bs>(k*bs,0)*iluvals.BLK<bs,bs>(pos*bs,0);
					}

					iluvals.BLK<bs,bs>(j*bs,0).noalias()
						= sum * iluvals.BLK<bs,bs>(diagind[bcolind[j]]*bs,0).inverse();
				}
				else
				{
					// compute u_ij
					iluvals.BLK<bs,bs>(j*bs,0) = data.BLK<bs,bs>(j*bs,0);

					for(index k = browptr[irow]; (k < browptr[irow+1]) && (bcolind[k] < irow); k++) 
					{
						index pos = -1;

						/* search for column index bcolind[j],
						 * between the diagonal index of row bcolind[k] 
						 * and the last index of row bcolind[k]
						 */
						inner_search(bcolind, 
							diagind[bcolind[k]], browptr[bcolind[k]+1], bcolind[j], &pos);

						if(pos == -1) continue;

						iluvals.BLK<bs,bs>(j*bs,0).noalias()
							-= iluvals.BLK<bs,bs>(k*bs,0)*iluvals.BLK<bs,bs>(pos*bs,0);
					}
				}
			}
		}
	}

	// invert diagonal blocks
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		iluvals.BLK<bs,bs>(diagind[irow]*bs,0) = iluvals.BLK<bs,bs>(diagind[irow]*bs,0).inverse();
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::precILUApply(const scalar *const r, 
                                              scalar *const __restrict z) const
{
	Eigen::Map<const Vector<scalar>> ra(r, nbrows*bs);
	Eigen::Map<Vector<scalar>> za(z, nbrows*bs);

	// No scaling like z := Sr done here
	
	/** solves Ly = Sr by asynchronous Jacobi iterations.
	 * Note that if done serially, this is a forward-substitution.
	 */
	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index i = 0; i < nbrows; i++)
		{
			Matrix<scalar,bs,1> sum = Matrix<scalar,bs,1>::Zero();

			for(index j = browptr[i]; j < diagind[i]; j++)
				sum += iluvals.BLK<bs,bs>(j*bs,0) * ytemp.SEG<bs>(bcolind[j]*bs);
			
			ytemp.SEG<bs>(i*bs) = ra.SEG<bs>(i*bs) - sum;
		}
	}

	/* Solves Uz = y by asynchronous Jacobi iteration.
	 * If done serially, this is a back-substitution.
	 */
	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index i = nbrows-1; i >= 0; i--)
		{
			Matrix<scalar,bs,1> sum = Matrix<scalar,bs,1>::Zero();

			for(index j = diagind[i]+1; j < browptr[i+1]; j++)
				sum += iluvals.BLK<bs,bs>(j*bs,0) * za.SEG<bs>(bcolind[j]*bs);
			
			za.SEG<bs>(i*bs) = iluvals.BLK<bs,bs>(diagind[i]*bs,0) * (ytemp.SEG<bs>(i*bs) - sum);
		}
	}

	// No correction of z needed because no scaling
}

template <typename scalar, typename index, int bs>
void BSRMatrix<scalar,index,bs>::printDiagnostic(const char choice) const
{
	std::ofstream fout("blockmatrix.txt");
	for(index i = 0; i < nbrows*bs; i++)
	{
		for(index j = 0; j < nbrows*bs; j++)
		{
			index brow = i/bs; index bcol = j/bs;
			int lcol = j%bs;
			bool found = false;
			for(index jj = browptr[brow]; jj < browptr[brow+1]; jj++)
				if(bcolind[jj] == bcol)
				{
					//fout << " " << vals[jj + lrow*bs + lcol];
					fout << " " << bcolind[jj]*bs+lcol+1;
					found = true;
					break;
				}
			if(!found)
				fout << " 0";
		}
		fout << '\n';
	}
	fout.close();
}



template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::BSRMatrix(const index n_brows,
		const index *const bcinds, const index *const brptrs,
		const short n_buildsweeps, const short n_applysweeps)
	: LinearOperator<scalar,index>('c'),vals(nullptr), isAllocVals(true), nbrows{n_brows}, 
	  dblocks(nullptr), iluvals(nullptr), scale(nullptr), ytemp(nullptr),
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{800}
{
	browptr = new index[nbrows+1];
	bcolind = new index[brptrs[nbrows]];
	diagind = new index[nbrows];
	vals = new scalar[brptrs[nbrows]];
	for(index i = 0; i < nbrows+1; i++)
		browptr[i] = brptrs[i];
	for(index i = 0; i < brptrs[nbrows]; i++)
		bcolind[i] = bcinds[i];

	// set diagonal blocks' locations
	for(index irow = 0; irow < nbrows; irow++) {
		for(index j = browptr[irow]; j < browptr[irow+1]; j++)
			if(bcolind[j] == irow) {
				diagind[irow] = j;
				break;
			}
	}
	
	std::cout << "BSRMatrix<1>: Set up CSR matrix with " << nbrows << " rows,\n    "
		<< nbuildsweeps << " build- and " << napplysweeps << " apply- async sweep(s)\n";
}

template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::BSRMatrix(const index nrows, const index *const brptrs,
		const index *const bcinds, const scalar *const values,
		const short n_buildsweeps, const short n_applysweeps)
	: LinearOperator<scalar,index>('c'),
	  vals(values), isAllocVals(false), bcolind(bcinds), browptr(brptrs), nbrows{nrows},
	  dblocks(nullptr), iluvals(nullptr), scale(nullptr), ytemp(nullptr),
	  nbuildsweeps{n_buildsweeps}, napplysweeps{n_applysweeps}, thread_chunk_size{800}
{
	// set diagonal blocks' locations
	for(index irow = 0; irow < nbrows; irow++) {
		for(index j = browptr[irow]; j < browptr[irow+1]; j++)
			if(bcolind[j] == irow) {
				diagind[irow] = j;
				break;
			}
	}
}

template <typename scalar, typename index>
BSRMatrix<scalar,index,1>::~BSRMatrix()
{
	if(isAllocVals)
		delete [] vals;
	if(bcolind)
		delete [] bcolind;
	if(browptr)
		delete [] browptr;
	if(diagind)
		delete [] diagind;
	if(dblocks)
		delete [] dblocks;
	if(iluvals)
		delete [] iluvals;
	if(ytemp)
		delete [] ytemp;
	if(scale)
		delete [] scale;

	bcolind = browptr = diagind = nullptr;
	vals = dblocks = iluvals = ytemp = scale = nullptr;
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::setAllZero()
{
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < browptr[nbrows]; i++)
		vals[i] = 0;
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::setDiagZero()
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		vals[diagind[irow]] = 0;
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::submitBlock(const index starti, const index startj,
		const scalar *const buffer, const index bsi, const index bsj)
{
	for(index irow = starti; irow < starti+bsi; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			if(bcolind[jj] < startj)
				continue;
			if(k == bsj) 
				break;
#ifdef DEBUG
			if(bcolind[jj] != startj+k)
				std::cout << "!  BSRMatrix<1>: submitBlock: Invalid block!!\n";
#endif
			vals[jj] = buffer[locrow*bsj+k];
			k++;
		}
	}
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::updateDiagBlock(const index starti,
		const scalar *const buffer, const index bs)
{
	// update the block, row-wise
	for(index irow = starti; irow < starti+bs; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = diagind[irow]-locrow; jj < diagind[irow]-locrow+bs; jj++)
		{
			index loccol = jj-diagind[irow]+locrow;
#pragma omp atomic update
			vals[jj] += buffer[locrow*bs+loccol];
			k++;
		}
	}
}

/** \warning It is assumed that locations corresponding to indices of entries to be added
 * already exist. If that's not the case, Bad Things (TM) will happen.
 */
template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::updateBlock(const index starti, const index startj,
		const scalar *const buffer, const index bsi, const index bsj)
{
	for(index irow = starti; irow < starti+bsi; irow++)
	{
		index k = 0;
		index locrow = irow-starti;
		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			if(bcolind[jj] < startj)
				continue;
			if(k == bsj) 
				break;
#ifdef DEBUG
			if(bcolind[jj] != startj+k)
				std::cout << "!  BSRMatrix<1>: updateBlock: Invalid block!!\n";
#endif
#pragma omp atomic update
			vals[jj] += buffer[locrow*bsi+k];
			k++;
		}
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::apply(const scalar a, const scalar *const xx,
                                       scalar *const __restrict yy) const
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		yy[irow] = 0;

		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			yy[irow] += a * vals[jj] * xx[bcolind[jj]];
		}
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::gemv3(const scalar a, const scalar *const __restrict__ xx, 
		const scalar b, const scalar *const yy, scalar *const zz) const
{
#pragma omp parallel for default(shared)
	for(index irow = 0; irow < nbrows; irow++)
	{
		zz[irow] = b * yy[irow];

		for(index jj = browptr[irow]; jj < browptr[irow+1]; jj++)
		{
			zz[irow] += a * vals[jj] * xx[bcolind[jj]];
		}
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precJacobiSetup()
{
	if(!dblocks) {
		dblocks = new scalar[browptr[nbrows]];
		std::cout << " CSR Matrix: precJacobiSetup(): Allocating.\n";
	}

#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		dblocks[irow] = 1.0/vals[diagind[irow]];
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precJacobiApply(const scalar *const rr, 
                                                 scalar *const __restrict zz) const
{
#pragma omp parallel for simd default(shared)
	for(index irow = 0; irow < nbrows; irow++)
		zz[irow] = dblocks[irow] * rr[irow];
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::allocTempVector()
{
	if(ytemp)
		std::cout << "!  BSRMatrix<1>: allocTempVector(): temp vector is already allocated!\n";
	else
		ytemp = new scalar[nbrows];
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precSGSApply(const scalar *const rr, 
                                              scalar *const __restrict zz) const
{
	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
		// forward sweep ytemp := D^(-1) (r - L ytemp)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < nbrows; irow++)
		{
			scalar inter = 0;

			for(index jj = browptr[irow]; jj < diagind[irow]; jj++)
				inter += vals[jj]*ytemp[bcolind[jj]];

			ytemp[irow] = dblocks[irow] * (rr[irow] - inter);
		}
	}

	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
		// backward sweep z := D^(-1) (D y - U z)

#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = nbrows-1; irow >= 0; irow--)
		{
			scalar inter = 0;
			
			// compute U z
			for(index jj = diagind[irow]+1; jj < browptr[irow+1]; jj++)
				inter += vals[jj] * zz[bcolind[jj]];

			// compute z = D^(-1) (D y - U z) for the irow-th block-segment of z
			zz[irow] = dblocks[irow] * ( vals[diagind[irow]]*ytemp[irow] - inter );
		}
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precILUSetup()
{
	if(!iluvals)
	{
		std::printf(" CSR Matrix: precILUSetup(): First-time setup\n");

		// Allocate lu
		iluvals = new scalar[browptr[nbrows]];
#pragma omp parallel for simd default(shared)
		for(int j = 0; j < browptr[nbrows]; j++) {
			iluvals[j] = vals[j];
		}

		// intermediate array for the solve part; NOT ZEROED
		if(!ytemp)
			ytemp = new scalar[nbrows];
		else
			std::cout << "! BSRMatrix<1>: precILUSetup(): Temp vector is already allocated!\n";
		
		if(!scale)
			scale = new scalar[nbrows];	
		else
			std::cout << "! BSRMatrix<1>: precILUSetup(): Scale vector is already allocated!\n";
	}

	// get the diagonal scaling matrix
	
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < nbrows; i++)
		scale[i] = 1.0/std::sqrt(vals[diagind[i]]);

	// compute L and U
	/** Note that in the factorization loop, the variable pos is initially set negative.
	 * If index is an unsigned type, that might be a problem. However,
	 * it should usually be okay as we are only comparing equality later.
	 */
	
	//printf("BSRMatrix<1>: precILUSetup: Factorizing. %d build sweeps, chunk size is %d.\n", 
	//		nbuildsweeps, thread_chunk_size);
	
	for(short isweep = 0; isweep < nbuildsweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index irow = 0; irow < nbrows; irow++)
		{
			for(index j = browptr[irow]; j < browptr[irow+1]; j++)
			{
				if(irow > bcolind[j])
				{
					scalar sum = scale[irow] * vals[j] * scale[bcolind[j]];

					for(index k = browptr[irow]; 
					    (k < browptr[irow+1]) && (bcolind[k] < bcolind[j]); 
					    k++  ) 
					{
						index pos = -1;
						inner_search<index> ( bcolind, 
							diagind[bcolind[k]], browptr[bcolind[k]+1], bcolind[j], &pos );

						if(pos == -1) {
							continue;
						}

						sum -= iluvals[k]*iluvals[pos];
					}

					iluvals[j] = sum / iluvals[diagind[bcolind[j]]];
				}
				else
				{
					// compute u_ij
					iluvals[j] = scale[irow]*vals[j]*scale[bcolind[j]];

					for(index k = browptr[irow]; (k < browptr[irow+1]) && (bcolind[k] < irow); k++) 
					{
						index pos = -1;

						/* search for column index bcolind[j], 
						 * between the diagonal index of row bcolind[k] 
						 * and the last index of row bcolind[k]
						 */
						inner_search(bcolind, 
							diagind[bcolind[k]], browptr[bcolind[k]+1], bcolind[j], &pos);

						if(pos == -1) continue;

						iluvals[j] -= iluvals[k]*iluvals[pos];
					}
				}
			}
		}
	}
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::precILUApply(const scalar *const __restrict ra, 
                                              scalar *const __restrict za) const
{
	// initially, z := Sr
#pragma omp parallel for simd default(shared)
	for(index i = 0; i < nbrows; i++) {
		za[i] = scale[i]*ra[i];
	}
	
	/** solves Ly = Sr by asynchronous Jacobi iterations.
	 * Note that if done serially, this is a forward-substitution.
	 */
	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index i = 0; i < nbrows; i++)
		{
			scalar sum = 0;
			for(index j = browptr[i]; j < diagind[i]; j++)
			{
				sum += iluvals[j] * ytemp[bcolind[j]];
			}
			ytemp[i] = za[i] - sum;
		}
	}

	/* Solves Uz = y by asynchronous Jacobi iteration.
	 * If done serially, this is a back-substitution.
	 */
	for(short isweep = 0; isweep < napplysweeps; isweep++)
	{
#pragma omp parallel for default(shared) schedule(dynamic, thread_chunk_size)
		for(index i = nbrows-1; i >= 0; i--)
		{
			scalar sum = 0;
			for(index j = diagind[i]+1; j < browptr[i+1]; j++)
			{
				sum += iluvals[j] * za[bcolind[j]];
			}
			za[i] = 1.0/iluvals[diagind[i]] * (ytemp[i] - sum);
		}
	}

	// correct z
#pragma omp parallel for simd default(shared)
	for(int i = 0; i < nbrows; i++)
		za[i] = za[i]*scale[i];
}

template <typename scalar, typename index>
void BSRMatrix<scalar,index,1>::printDiagnostic(const char choice) const
{
	std::ofstream fout("pointmatrix.txt");
	for(index i = 0; i < nbrows; i++)
	{
		for(index j = 0; j < nbrows; j++)
		{
			bool found = false;
			for(index jj = browptr[i]; jj < browptr[i+1]; jj++)
				if(bcolind[jj] == j)
				{
					fout << " " << bcolind[jj]+1;
					found = true;
					break;
				}
			if(!found)
				fout << " 0";
		}
		fout << '\n';
	}
	fout.close();
}

