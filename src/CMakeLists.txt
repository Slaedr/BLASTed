# For some reason, CMake can't figure out PIC flag for Cray
if(CXX_COMPILER_CRAY)
	add_compile_options(-fPIC)
endif()

add_library(helper helper_algorithms.cpp)
set_property(TARGET helper PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(rawmatrixutils rawsrmatrixutils.cpp)
set_property(TARGET rawmatrixutils PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(orderingscaling reorderingscaling.cpp)
set_property(TARGET orderingscaling PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(blockmatrices csrmatrix.cpp bsrmatrix.cpp matvecs.cpp scmatrix.cpp)
set_property(TARGET blockmatrices PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(blockmatrices rawmatrixutils)

add_library(coomatrix coomatrix.cpp)
target_link_libraries(coomatrix blockmatrices)
set_property(TARGET coomatrix PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(solverops
  solverfactory.cpp
  relaxation_jacobi.cpp relaxation_chaotic.cpp relaxation_async_sgs.cpp
  solverops_jacobi.cpp solverops_sgs.cpp solverops_ilu0.cpp solverops_base.cpp
  async_blockilu_factor.cpp async_ilu_factor.cpp)
set_property(TARGET solverops PROPERTY POSITION_INDEPENDENT_CODE ON)
if(CXX_COMPILER_CLANG)
  target_compile_options(solverops PRIVATE "-Wno-error=pass-failed")
endif()

if(WITH_PETSC)

	add_library(blasted_petsc SHARED blasted_petsc.cpp)
	target_link_libraries(blasted_petsc solverops orderingscaling rawmatrixutils helper
	  ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})
	set_property(TARGET blasted_petsc PROPERTY POSITION_INDEPENDENT_CODE ON)
	
	add_executable(util_petsc_io util_petsc_io.c blasted_petsc_io.cpp)
	target_link_libraries(util_petsc_io coomatrix blockmatrices ${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})
	
endif()

# Tags

set_source_files_properties(tags PROPERTIES GENERATED true)
add_custom_target ( tags
    COMMAND ctags -R --c++-kinds=+pf --extra=+q . 
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} 
  )

find_program(CTAGS_PATH ctags)
if(CTAGS_PATH)
    message(STATUS "Found ctags: ${CTAGS_PATH}")
else()
	message(STATUS "ctags not found! Tags will not be generated.")
endif(CTAGS_PATH)


