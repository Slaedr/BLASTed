cmake_minimum_required(VERSION 3.0)

# Variables that can/must be passed to the cmake command:
#
# - The environment variable EIGEN_DIR needs to be set to the path of the Eigen top-level directory.
# - -DNOOMP=1 to compile without OpenMP
# - -DSSE=1 to compile with SSE 4.2 instructions; ignored when compiling for KNC.
# - -DAVX=1 to compile with AVX instructions.
# - -DSKYLAKE=1 to compile with AVX-512 instructions for Xeon Skylake CPUs
# - -DKNL=1 to compile for Xeon Phi Knights Landing
# - -DMICKNC=1 to compile for Xeon Phi Knights Corner
# - -DWITH_PETSC=1 to compile with PETSc support; in which case
#     make sure PETSC_LIB and PETSC_ARCH envonment variables have been set.
#     Note that in this case, MPI is required.
# - -DPROFILE=1 for profiling with gprof.

project (blasted)

set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -Werror")
set (CMAKE_C_FLAGS "-std=c11 -Wall -Werror")
set (CMAKE_CXX_FLAGS_DEBUG " ")
set (CMAKE_CXX_FLAGS_RELEASE " ")

# to compile with OpenMP
if(NOOMP)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hnoomp")
	endif()
	message(STATUS "Compiling without OpenMP")
else()
	if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
		message(STATUS "Compiling with OpenMP for GNU")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
		message(STATUS "Compiling with OpenMP for Intel")
	elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
		message(STATUS "Compiling with OpenMP for Cray")
	else()
		message(SEND_ERROR "Don't know OpenMP flag for this compiler")
	endif()
endif()

# set compile options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-inline-vec-optimized-missed=optimizations.info")
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass=inline,vectorize")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=2")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hlist=a")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1")
endif()

if(MICKNC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
	message(STATUS "Compiling for KNC")
	if(NOT("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
		message(SEND_ERROR "Need to use Intel compiler for Xeon Phi!")
	endif()
	
	# Link the mic version of MKL if requested
	if(WITH_MKL)
		include_directories($ENV{MKL_ROOT}/include)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MKL -L${MKLROOT}/lib/mic -lmkl_rt -lpthread -lm -ldl")
		message(STATUS "Linking MKL for mic.")
	endif()
else()
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
	message(STATUS "Compiling for x86_64")
	
	# Select hardware-specific optimizations
	if(SSE)
		set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse4.2")
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with SSE4.2")
		endif()
	elseif(AVX)
		set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx")
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX")
		endif()
	elseif(SKYLAKE)
		if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xcore-avx512")
		elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=skylake-avx512")
		else()
			message(SEND_ERROR "Don't know/require option for this compiler!")
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX-512 for Skylake")
		endif()
	elseif(KNL)
		if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xmic-avx512")
		elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=knl")
		else()
			message(SEND_ERROR "Don't know/require option for this compiler!")
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX-512 for KNL")
		endif()
	endif()

	# Link MKL if requested
	if(WITH_MKL)
		include_directories($ENV{MKL_ROOT}/include)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MKL -L$ENV{MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl")
		message(STATUS "Linking MKL.")
	endif()
endif()

# profiling
if(PROFILE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

# Eigen
include_directories($ENV{EIGEN_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")

# PETSc
if(WITH_PETSC)
	find_library(PETSC_LIB NAMES petsc PATHS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
	include_directories($ENV{PETSC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)
	message(STATUS "Building with PETSc found at ${PETSC_LIB}")
endif()

# ---------------------------------------------------------------------------- #

# flags and stuff

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -UDEBUG -DNDEBUG")
include_directories(${CMAKE_SOURCE_DIR}/include)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode")
else()
	message(STATUS "Release mode")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# libraries to be compiled

add_subdirectory(src)

# Tests
enable_testing()
add_subdirectory(tests)
