/** \file iterative_preconditioner.hpp
 * \brief Utility code for iterative preconditioners
 */

#include "iterative_preconditioner.hpp"

namespace blasted {

BuildIterParams extractBuildIterParams(const IterPrecParams params)
{
	BuildIterParams bip;
	bip.usescaling = params.usescaling;
	bip.thread_chunk_size = params.thread_chunk_size;
	bip.threaded = params.threadedfactor;
	bip.nsweeps = params.nbuildsweeps;
	bip.inittype = params.factinittype;
	bip.itertype = params.buildtype;
	return bip;
}

ApplyIterParams extractApplyIterParams(const IterPrecParams params)
{
	ApplyIterParams aip;
	aip.usescaling = params.usescaling;
	aip.thread_chunk_size = params.thread_chunk_size;
	aip.threaded = params.threadedapply;
	aip.nsweeps = params.napplysweeps;
	aip.inittype = params.applyinittype;
	aip.itertype = params.applytype;
	return aip;
}

}
