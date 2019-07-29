/** \file
 * \brief Specification of diagnostic data
 */

#ifndef BLASTED_PRECONDITIONER_DIAGNOSTICS_H
#define BLASTED_PRECONDITIONER_DIAGNOSTICS_H

#include <array>
#include <vector>

namespace blasted {

/// Information that may be computed by a preconditioner for aiding analysis
struct PrecInfo
{
	double prec_remainder_norm;   ///< Difference norm between computed preconditioning matrix and ideal
	double prec_rem_initial_norm; ///< Difference norm between initial guess matrix and ideal
	double upper_min_diag_dom;    ///< Minimum diagonal dominance in upper factor, if any
	double upper_avg_diag_dom;    ///< Average diagonal dominance in upper factor, if any
	double lower_min_diag_dom;    ///< Minimum diagonal dominance in lower factor, if any
	double lower_avg_diag_dom;    ///< Average diagonal dominance in lower factor, if any
};

/// Information about a preconditioner over a sequence of linear solves
/** Static members are initialized in src/solverfactory.cpp
 */
struct PrecInfoList
{
	std::vector<PrecInfo> infolist;

	/// Strings describing the quantities computed, in the same order as \ref PrecInfo
	static const std::array<std::string,6> descr;

	/// Text width required for writing each string in descr
	static const int field_width;
};

}

#endif
