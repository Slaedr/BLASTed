/** \file
 * \brief Specification of diagnostic data
 */

#ifndef BLASTED_PRECONDITIONER_DIAGNOSTICS_H
#define BLASTED_PRECONDITIONER_DIAGNOSTICS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
	double prec_remainder_norm;   ///< Difference norm between computed preconditioning matrix and ideal
	double upper_min_diag_dom;    ///< Minimum diagonal dominance in upper factor, if any
	double upper_avg_diag_dom;    ///< Average diagonal dominance in upper factor, if any
	double lower_min_diag_dom;    ///< Minimum diagonal dominance in lower factor, if any
	double lower_avg_diag_dim;    ///< Average diagonal dominance in lower factor, if any
} BlastedPrecDiagnostics;

#ifdef __cplusplus
}
#endif

#endif
