mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DWITH_PETSC=1
