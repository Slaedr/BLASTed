/** \file error_handling.hpp
 * \brief Exceptions etc.
 */

#include <stdexcept>

#ifndef BLASTED_ERROR_HANDLING_H
#define BLASTED_ERROR_HANDLING_H

namespace blasted {

class PetscException : public std::runtime_error
{
public:
	//PetscException(const char *msg) : std::runtime_error(msg);
	PetscException(const std::string msg) : std::runtime_error(msg) { }
};

class PetscOptionNotFound : public PetscException
{
public:
	PetscOptionNotFound(const std::string msg) : PetscException(msg) { }
};

class InvalidPetscOption: public PetscException
{
public:
	InvalidPetscOption(const std::string msg) : PetscException(msg) { }
};

}

#endif
