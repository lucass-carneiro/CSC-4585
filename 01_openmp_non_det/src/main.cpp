/**
 * This program is a minimal OpenMP example which demonstrates how to create threads and
 * non-deerministic execution
 */

#include <fmt/base.h>
#include <omp.h>

auto main(int, char **) -> int {
  fmt::print("This is how parallel programmers order the elemets of an array [ ");

#pragma omp parallel default(none)
  {
    fmt::print("{} ", omp_get_thread_num());
  }

  fmt::println("]");

  return 0;
}