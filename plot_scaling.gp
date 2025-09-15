set key top left

set title "Speedups"

set xlabel "Number of threads"
set ylabel "Speedup"

plot "openmp_pi_scaling.dat" using 1:3 with linespoints title "std::vector", \
     "openmp_pi_critical_scaling.dat" using 1:3 with linespoints title "omp critical", \
     "openmp_pi_parallel_for_scaling.dat" using 1:3 with linespoints title "parallel for reduction",

pause -1 "Press Enter to continue"