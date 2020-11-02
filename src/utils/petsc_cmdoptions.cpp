/** \file petsc_cmdoptions.cpp
 * \brief Parse options from different sources.
 * \author Aditya Kashi
 * \date 2017-10
 */

#include "petsc_cmdoptions.hpp"
#include "utils/mpiutils.hpp"
#include <iostream>
#include <cstdlib>
#include <petscsys.h>

namespace blasted {

typedef int StatusCode;

std::string parsePetscErrorCode(const int ierr)
{
	if(ierr <= PETSC_ERR_MIN_VALUE)
		throw std::runtime_error("Invalid PETSc error code!");
	else if(ierr >= PETSC_ERR_MAX_VALUE)
		throw std::runtime_error("Invalid PETSc error code!");

	const char *text; char *specific;
	int jerr = PetscErrorMessage(ierr, &text, &specific);
	if (jerr != 0)
		throw std::runtime_error("Could not get PETSc error message!");
	if(!text)
		throw std::runtime_error("PETSc error text is null!");

	std::string msg = text;
	if(specific) {
		msg += " ";
		std::string spec = specific;
		msg += spec;
	}

	return msg;
}

Petsc_exception::Petsc_exception(const int ierr) 
	: std::runtime_error(std::string("PETSc error: ") + parsePetscErrorCode(ierr))
{ }

InputNotGivenError::InputNotGivenError(const std::string& msg) 
	: std::runtime_error(std::string("Input not given: ")+msg)
{ }

bool parsePetscCmd_isDefined(const std::string optionname)
{
	StatusCode ierr = 0;
	PetscBool flg = PETSC_FALSE;
	ierr = PetscOptionsHasName(NULL, NULL, optionname.c_str(), &flg); petsc_throw(ierr);
	
	if(flg == PETSC_TRUE)
		return true;
	else
		return false;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
int parsePetscCmd_int(const std::string optionname)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	int output = 0;
	ierr = PetscOptionsGetInt(NULL, NULL, optionname.c_str(), &output, &set); petsc_throw(ierr);
	if(!set) {
		throw InputNotGivenError(std::string("Integer ") + optionname + std::string(" not set"));
	}
	return output;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
PetscReal parseOptionalPetscCmd_real(const std::string optionname, const PetscReal defval)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	PetscReal output = 0;
	ierr = PetscOptionsGetReal(NULL, NULL, optionname.c_str(), &output, &set); petsc_throw(ierr);
	if(!set) {
		std::cout << "PETSc cmd option " << optionname << " not set; using default.\n";
		output = defval;
	}
	return output;
}

bool parsePetscCmd_bool(const std::string optionname)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	PetscBool output = PETSC_FALSE;
	ierr = PetscOptionsGetBool(NULL, NULL, optionname.c_str(), &output, &set); petsc_throw(ierr);
	if(!set) {
		throw InputNotGivenError(std::string("Boolean ") + optionname + std::string(" not set"));
	}
	return (bool)output;
}


std::string parsePetscCmd_string(const std::string optionname, const size_t p_strlen)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	char* tt = new char[p_strlen+1];
	ierr = PetscOptionsGetString(NULL, NULL, optionname.data(), tt, p_strlen, &set); petsc_throw(ierr);
	if(!set) {
		throw InputNotGivenError(std::string("String ") + optionname + std::string(" not set"));
	}
	const std::string stropt = tt;
	delete [] tt;
	return stropt;
}

/** Ideally, we would have single function template for int and real, but for that we need
 * `if constexpr' from C++ 17 which not all compilers have yet.
 */
std::vector<int> parsePetscCmd_intArray(const std::string optionname, const int maxlen)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	std::vector<int> arr(maxlen);
	int len = maxlen;

	ierr = PetscOptionsGetIntArray(NULL, NULL, optionname.c_str(), &arr[0], &len, &set);
	petsc_throw(ierr);
	arr.resize(len);

	if(!set) {
		throw InputNotGivenError(std::string("Array ") + optionname + std::string(" not set"));
	}
	return arr;
}

std::vector<int> parseOptionalPetscCmd_intArray(const std::string optionname, const int maxlen)
{
	StatusCode ierr = 0;
	PetscBool set = PETSC_FALSE;
	std::vector<int> arr(maxlen);
	int len = maxlen;

	ierr = PetscOptionsGetIntArray(NULL, NULL, optionname.c_str(), &arr[0], &len, &set);
	petsc_throw(ierr);
	arr.resize(len);

	if(!set) {
		throw InputNotGivenError(std::string("Array ") + optionname + std::string(" not set"));
	}
	return arr;
}

}
