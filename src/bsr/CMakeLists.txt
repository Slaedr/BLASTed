# For some reason, CMake can't figure out PIC flag for Cray
if(CXX_COMPILER_CRAY)
	add_compile_options(-fPIC)
endif()



add_library(rawmatrixutils rawsrmatrixutils.cpp scmatrix.cpp)
set_property(TARGET rawmatrixutils PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(orderingscaling reorderingscaling.cpp)
set_property(TARGET orderingscaling PROPERTY POSITION_INDEPENDENT_CODE ON)
if(WITH_MC64)
  target_link_libraries(orderingscaling ${MC64_LIB} ${FORTLDFLAGS})
endif(WITH_MC64)

add_library(srblockmatrices csrmatrix.cpp bsrmatrix.cpp matvecs.cpp)
set_property(TARGET srblockmatrices PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(srblockmatrices rawmatrixutils)

add_library(srsolverops
  solverfactory.cpp
  solverops_levels_sgs.cpp solverops_levels_ilu0.cpp
  relaxation_chaotic.cpp
  solverops_jacobi.cpp solverops_sgs.cpp solverops_ilu0.cpp solverops_base.cpp
  async_blockilu_factor.cpp async_ilu_factor.cpp
  ilu_pattern.cpp levelschedule.cpp ../blas1.cpp
  )
set_property(TARGET srsolverops PROPERTY POSITION_INDEPENDENT_CODE ON)
if(CXX_COMPILER_CLANG)
  target_compile_options(srsolverops PRIVATE "-Wno-error=pass-failed")
endif()
target_link_libraries(srsolverops orderingscaling helper)

