mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DAVX=1 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DWITH_PETSC=1
