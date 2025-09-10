/**
 * This program is a minimal OpenMP example.
 */

#include <fmt/base.h>
#include <omp.h>

auto main(int, char **) -> int {

#pragma omp parallel default(none)
  {
    fmt::print("hello");
    fmt::print("world \n");
  }

  return 0;
}