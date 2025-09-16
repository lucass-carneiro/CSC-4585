# CSC-4585
Course Material for CSC 4585 at LSU

# Compiling

1. Install and bootstrap `vcpkg`. See [here](https://www.mpich.org/static/docs/v4.3.0/www3/MPI_Barrier.html) for details.
2. Export the `VCPKG_ROOT` env. variable and add `vcpkg` to your path.
3. Build with `cmake --preset "x64-debug-vcpkg" -S .` and `cmake --build build-x64-debug-vcpkg -j2`

# Scaling test plots

Once scaling test files are produced, run `gnuplot plot_scaling.gp`