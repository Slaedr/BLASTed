
add_library(myblas matvecs.cpp blas1.cpp)
set_property(TARGET myblas PROPERTY POSITION_INDEPENDENT_CODE ON)
