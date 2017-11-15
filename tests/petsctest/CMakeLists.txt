# test executables

add_executable(testcsr_poisson poisson3d.cpp)
target_link_libraries(testcsr_poisson blasted_petsc)

