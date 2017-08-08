cmake_minimum_required(VERSION 2.8)

# Required arguments:
# The environment variable EIGEN_DIR needs to be set to the path where
# the Eigen source directory is present.

# Optional arguments:
# Pass -DOMP=1 to compile with OpenMP
# Pass -DMICKNC=1 to compile for Xeon Phi Knights Corner.
# Pass -DNOAVX=1 to compile with SSE 4.2 instructions rather than AVX (default) instructions;
#                 ^ ignored when compiling for KNC.
# Pass -DWITH_MKL=1 to build with Intel Math Kernel Library

project (blasted)

# to compile with OpenMP
if(OMP)
	message(STATUS "Compiling with OpenMP")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
	else()
		message(SEND_ERROR "Don't know OpenMP flag for this compiler")
	endif()
endif()

# set compile options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ftree-vectorizer-verbose=2")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -qopt-report=2")
endif()

# Eigen
include_directories($ENV{EIGEN_DIR})

if(MICKNC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
	message(STATUS "Compiling for KNC")
	if(NOT("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
		message(SEND_ERROR "Need to use Intel compiler for Xeon Phi!")
	endif()
	
	if(WITH_MKL)
		include_directories($ENV{MKL_ROOT}/include)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MKL -L${MKLROOT}/lib/mic -lmkl_rt -lpthread -lm -ldl")
		message(STATUS "Linking MKL.")
	endif()
else()

	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
	
	if(NOAVX)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
	else()
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
	endif()
	
	if(WITH_MKL)
		#find_library(MKL_LIB NAMES mkl_rt PATHS $ENV{MKL_ROOT}/lib/intel64)
		include_directories($ENV{MKL_ROOT}/include)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MKL -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl")
		message(STATUS "Linking MKL.")
	endif()
endif()

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC)

# ---------------------------------------------------------------------------- #

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3 -UDEBUG -DNDEBUG")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode")
else()
	message(STATUS "Release mode")
endif()

#set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# libraries to be compiled
#add_library(blasted STATIC blockmatrices.cpp)


