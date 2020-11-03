/** \file cmdoptions.cpp
 * \brief Implementation of command-line option parsing using Boost
 */

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/parsers.hpp>

namespace blasted {

namespace po = boost::program_options;

SolverSettings parse_cmd_options(const int argc, const char *const argv[])
{
	po::options_description desc (std::string("BLASTed"));

	desc.add_options()
		("help", "BLASTed solver")
		("blasted_pc_type", po::value<std::string>(),
		 "Type of preconditioner: jacobi, gs, sgs, chaotic, ilu0")
		("blasted_pc_build_iter_type", po::value<std::string>(),
		 "Type of iteration to use for building the preconditioner: jacobi, async or gauss_seidel")
		("blasted_pc_apply_iter_type", po::value<std::string>(),
		 "Type of iteration to use for applying the preconditioner: jacobi, async or gauss_seidel");

	po::variables_map cmdvarmap;
	po::parsed_options parsedopts
		= po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
	po::store(parsedopts, cmdvarmap);
	po::notify(cmdvarmap);

	SolverSettings sset;

	sset.prectype = solverTypeFromString(cmdvarmap["blasted_pc_type"].as<std::string>());

	return cmdvarmap;
}

}
