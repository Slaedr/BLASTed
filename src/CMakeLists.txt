# For some reason, CMake can't figure out PIC flag for Cray
if(CXX_COMPILER_CRAY)
  add_compile_options(-fPIC)
endif()

add_subdirectory(blas)

add_library(helper helper_algorithms.cpp)
set_property(TARGET helper PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(rawmatrixutils rawsrmatrixutils.cpp adjacency.cpp scmatrix.cpp)
set_property(TARGET rawmatrixutils PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(orderingscaling reorderingscaling.cpp)
set_property(TARGET orderingscaling PROPERTY POSITION_INDEPENDENT_CODE ON)
if(WITH_MC64)
  target_link_libraries(orderingscaling ${MC64_LIB} ${FORTLDFLAGS})
endif(WITH_MC64)

add_library(blockmatrices csrmatrix.cpp bsrmatrix.cpp)
set_property(TARGET blockmatrices PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(blockmatrices myblas rawmatrixutils)

add_library(coomatrix coomatrix.cpp)
target_link_libraries(coomatrix blockmatrices)
set_property(TARGET coomatrix PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(solverops
  solverfactory.cpp
  sai.cpp
  solverops_sai.cpp
  solverops_levels_sgs.cpp solverops_levels_ilu0.cpp
  relaxation_chaotic.cpp
  solverops_jacobi.cpp solverops_sgs.cpp solverops_ilu0.cpp solverops_base.cpp
  async_blockilu_factor.cpp async_ilu_factor.cpp
  ilu_pattern.cpp levelschedule.cpp matrix_properties.cpp
  )
set_property(TARGET solverops PROPERTY POSITION_INDEPENDENT_CODE ON)
if(CXX_COMPILER_CLANG)
  target_compile_options(solverops PRIVATE "-Wno-error=pass-failed")
endif()
target_link_libraries(solverops myblas orderingscaling helper)

if(WITH_PETSC)

  add_library(blasted_petsc SHARED blasted_petsc.cpp)

  target_link_libraries(blasted_petsc solverops orderingscaling rawmatrixutils helper
	${PETSC_LIB} ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})

  set_property(TARGET blasted_petsc PROPERTY POSITION_INDEPENDENT_CODE ON)
  
endif()

add_subdirectory(utils)

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


