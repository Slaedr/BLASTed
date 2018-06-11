cmake_minimum_required(VERSION 3.0)

# Variables that can/must be passed to the cmake command:
#
# - The environment variable EIGEN_DIR needs to be set to the path of the Eigen top-level directory.
#
# - -DWITH_PETSC=1 to compile with PETSc support; in which case
#     make sure PETSC_LIB and PETSC_ARCH envonment variables have been set.
#     Note that in this case, MPI is required.
# - -DNOOMP=1 to compile without OpenMP
# - -DSSE=1 to compile with SSE 4.2 instructions; ignored when compiling for KNC.
# - -DAVX=1 to compile with AVX instructions.
# - -DBROADWELL=1 to compile for Intel Xeon Broadwell CPUs.
# - -DSKYLAKE=1 to compile with AVX-512 instructions for Xeon Skylake CPUs.
# - -DKNL=1 to compile for Xeon Phi Knights Landing
# - -DMICKNC=1 to compile for Xeon Phi Knights Corner (deprecated).
# - -DPROFILE=1 for profiling with gprof.
# - -DSLURM=1 for running the automated tests on a system managed by SLURM
#
# - -DBUILD_BLOCK_SIZE=<n> for some integer n, to build the block solver operations for
#     an additional block size of <n>. By default, block sizes 4 and 5 are supported.

project (blasted)

# detect C++ compiler - the C compiler must be the same type too
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CXX_COMPILER_GNUCLANG TRUE)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	set(CXX_COMPILER_GNU TRUE)
	message(STATUS "GNU C++ compiler detected")
	if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "GNU"))
	  message(SEND_ERROR "C and C++ compilers should match!")
	endif()
  else()
	set(CXX_COMPILER_CLANG TRUE)
	message(STATUS "Clang C++ compiler detected")
	if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "Clang"))
	  message(SEND_ERROR "C and C++ compilers should match!")
	endif()
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
	set(CXX_COMPILER_INTEL TRUE)
	message(STATUS "Intel C++ compiler detected")
	if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "Intel"))
		message(SEND_ERROR "C and C++ compilers should match!")
	endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Cray")
	set(CXX_COMPILER_CRAY TRUE)
	message(STATUS "Cray C++ compiler detected")
	if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "Cray"))
		message(SEND_ERROR "C and C++ compilers should match!")
	endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "PGI")
	set(CXX_COMPILER_PGI TRUE)
	message(STATUS "PGI C++ compiler detected")
	if(NOT("${CMAKE_C_COMPILER_ID}" MATCHES "PGI"))
		message(SEND_ERROR "C and C++ compilers should match!")
	endif()
else()
	message(SEND_ERROR "Could not identify compiler!")
endif()

# Basic options
if(CXX_COMPILER_GNUCLANG OR CXX_COMPILER_INTEL)
	set (CMAKE_CXX_FLAGS "-std=c++14 -Wall -Werror")
	set (CMAKE_C_FLAGS "-std=c11 -Wall -Werror")
	set (CMAKE_CXX_FLAGS_DEBUG "-g")
	set (CMAKE_CXX_FLAGS_RELEASE "-O3")
elseif(CXX_COMPILER_PGI)
	set (CMAKE_CXX_FLAGS "-std=c++14")
	set (CMAKE_C_FLAGS "-std=c11")
	set (CMAKE_CXX_FLAGS_DEBUG "-g")
	set (CMAKE_CXX_FLAGS_RELEASE "-O2")
elseif(CXX_COMPILER_CRAY)
	set (CMAKE_CXX_FLAGS "-h std=c++14 -h error_on_warning")
	set (CMAKE_C_FLAGS "-h std=c11 -h error_on_warning")
	set (CMAKE_CXX_FLAGS_DEBUG "-g -O1")
	set (CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

# to compile with OpenMP
if(NOOMP)
	if(CXX_COMPILER_GNUCLANG)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
		set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
	endif()
	if(CXX_COMPILER_PGI)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nomp")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -nomp")
	endif()
	if(CXX_COMPILER_CRAY)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hnoomp")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -hnoomp")
	endif()
	message(STATUS "Compiling without OpenMP")
else()
	if(CXX_COMPILER_GNUCLANG)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
		message(STATUS "Compiling with OpenMP for GNU")
	elseif(CXX_COMPILER_INTEL)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
		set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")	  
		message(STATUS "Compiling with OpenMP for Intel")
	elseif(CXX_COMPILER_CRAY)
		message(STATUS "Compiling with OpenMP for Cray")
	elseif(CXX_COMPILER_PGI)
		message(STATUS "Compiling with OpenMP for PGI")
	else()
		message(SEND_ERROR "Don't know OpenMP flag for this compiler!")
	endif()
endif()

# set verbosity options
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-inline-vec-optimized-missed=optimizations.info")
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass=inline,vectorize")
elseif(CXX_COMPILER_INTEL)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=2")
elseif(CXX_COMPILER_CRAY)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -hlist=a -hreport=fisv")
endif()

# Architecture to target
if(MICKNC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmic")
	message(STATUS "Compiling for KNC")
	if(NOT CXX_COMPILER_INTEL)
		message(SEND_ERROR "Need to use Intel compiler for Xeon Phi!")
	endif()
	
	# Link the mic version of MKL if requested
	if(WITH_MKL)
		include_directories($ENV{MKL_ROOT}/include)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_MKL -L${MKLROOT}/lib/mic -lmkl_rt -lpthread -lm -ldl")
		message(STATUS "Linking MKL for mic.")
	endif()
else()
	# Select hardware-specific optimizations
	if(AVX)
		if(CXX_COMPILER_GNUCLANG OR CXX_COMPILER_INTEL)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx")
		elseif(CXX_COMPILER_PGI)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=sandybridge")
		elseif(CXX_COMPILER_CRAY)
			add_definitions(-DEIGEN_VECTORIZE_AVX)
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX")
		endif()
	elseif(BROADWELL)
		if(CXX_COMPILER_INTEL)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xcore-avx2")
		elseif(CXX_COMPILER_GNUCLANG)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=broadwell")
		elseif(CXX_COMPILER_PGI)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=haswell")
		elseif(CXX_COMPILER_CRAY)
			add_definitions(-DEIGEN_VECTORIZE_AVX2 -DEIGEN_VECTORIZE_FMA)
		else()
			message(SEND_ERROR "Don't know/require option for this compiler!")
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX-2 for Broadwell")
		endif()
	elseif(SKYLAKE)
		if(CXX_COMPILER_INTEL)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xcore-avx512")
		elseif(CXX_COMPILER_GNUCLANG)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=skylake-avx512")
		elseif(CXX_COMPILER_PGI)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=skylake")
		elseif(CXX_COMPILER_CRAY)
			message(STATUS "Architecture options for Cray are to be set by the compiler wrapper.")
			# But we need to tell Eigen something
			add_definitions(-DEIGEN_VECTORIZE_AVX512 -DEIGEN_VECTORIZE_FMA)
		else()
			message(SEND_ERROR "Don't know option for this compiler!")
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX-512 for Skylake")
		endif()
	elseif(KNL)
		if(CXX_COMPILER_INTEL)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xmic-avx512")
		elseif(CXX_COMPILER_GNUCLANG)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=knl")
		elseif(CXX_COMPILER_PGI)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=knl")
		else()
			message(SEND_ERROR "Don't know/require option for this compiler!")
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with AVX-512 for KNL")
		endif()
	else()
		if(CXX_COMPILER_GNUCLANG OR CXX_COMPILER_INTEL)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse4.2")
		elseif(CXX_COMPILER_PGI)
			set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -tp=x64,nehalem")
		endif()
		if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
			message(STATUS "Compiling with SSE4")
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
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g")
endif()

# Boost
find_package(Boost 1.60 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Eigen
include_directories($ENV{EIGEN_DIR})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE")
	
if(SLURM)
	set(SEQEXEC "srun")
	set(SEQTASKS "-n 1")
else()
	set(SEQEXEC "")
	set(SEQTASKS "")
endif(SLURM)

# PETSc
if(WITH_PETSC)
	find_package(MPI REQUIRED)
	#message(STATUS "Found MPI include: ${MPI_C_INCLUDE_PATH}, ${MPI_CXX_INCLUDE_PATH}")
	#message(STATUS "Found MPI libraries: ${MPI_C_COMPILE_FLAGS}, ${MPI_C_LINK_FLAGS}, ${MPI_C_LIBRARIES}")
	include_directories(${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
	if(SLURM)
		set(MPIEXEC "srun")
	else()
		set(MPIEXEC "mpirun")
	endif(SLURM)

	find_library(PETSC_LIB NAMES petsc PATHS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
	include_directories($ENV{PETSC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)
	message(STATUS "Building with PETSc found at ${PETSC_LIB}")
endif()

# ---------------------------------------------------------------------------- #

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -UDEBUG -DNDEBUG")
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
