/**
 * This program computes the value of pi by integrating the function
 * $$ \int_{0}^{1} \frac{4}{\sqrt{1 + x^2}} dx $$
 *
 * To do that, we partition the interval [0, 1] into n parallelograms that approximate the function
 * and sum their areas serially.
 *
 * We give each participating thread an equal number of parallelograms to work on and combine
 * results at the end.
 *
 * We compare the final result with C++20's std::numbers::pi
 */
#include <argparse/argparse.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fmt/base.h>
#include <numbers>
#include <omp.h>
#include <vector>

auto integrand(double x) -> double { return 4.0 / (1.0 + x * x); }

auto main(int argc, char **argv) -> int {
  // Argument handling
  argparse::ArgumentParser program("serial_pi");

  using num_blocks_t = std::uint64_t;
  constexpr auto num_blocks_arg_str = "num_blocks";

  program.add_argument(num_blocks_arg_str)
      .help("Number of blocks to use for the integration")
      .required()
      .scan<'u', num_blocks_t>();

  using num_threads_t = int;
  constexpr auto num_threads_arg_string = "num_threads";

  program.add_argument(num_threads_arg_string)
      .help("Number of threads to use when integrating")
      .required()
      .scan<'i', num_threads_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    fmt::println("CLI error: {}", err.what());
    return EXIT_FAILURE;
  }

  const auto num_blocks = program.get<num_blocks_t>(num_blocks_arg_str);
  const auto num_threads = program.get<num_threads_t>(num_threads_arg_string);

  // Partitioning the interval
  fmt::println("Computing pi using {} blocks", num_blocks);

  const double interval_start = 0.0;
  const double interval_end = 1.0;
  const auto interval_step = (interval_end - interval_start) / static_cast<double>(num_blocks);

  // *Request* a numeber of threads to use and begin parallel region
  omp_set_num_threads(num_threads);

  // Hold the areas computed on each thread
  std::vector<double> thread_areas(static_cast<std::size_t>(num_threads));
  thread_areas.reserve(static_cast<std::size_t>(num_threads));

  const auto compute_start_time = std::chrono::steady_clock::now();

  // Launch threads and compute areas
#pragma omp parallel default(none) shared(thread_areas)                                            \
    firstprivate(num_blocks, num_threads, interval_step)
  {
    const auto actual_num_threads = static_cast<std::uint64_t>(omp_get_num_threads());
    const auto thread_id = static_cast<std::uint64_t>(omp_get_thread_num());

    const auto blocks_per_thread = num_blocks / actual_num_threads;

    const auto interval_start
        = static_cast<double>(thread_id) * 1.0 / static_cast<double>(actual_num_threads);

    double thread_area = 0;

    for (std::uint64_t i = 0; i < blocks_per_thread; i++) {
      const auto x0 = interval_start + static_cast<double>(i) * interval_step;
      const auto x1 = x0 + interval_step;

      const auto y0 = integrand(x0);
      const auto y1 = integrand(x1);

      const auto tallest{y0 > y1 ? y0 : y1};
      const auto shortest{y0 < y1 ? y0 : y1};

      const auto rect_area = interval_step * shortest;
      const auto tri_area = interval_step * (tallest - shortest) / 2.0;

      thread_area += rect_area + tri_area;
    }

    thread_areas[thread_id] = thread_area;
  }

  // Summ all areas
  double total_area = 0.0;
  for (const auto &area : thread_areas) {
    total_area += area;
  }

  const auto compute_end_time = std::chrono::steady_clock::now();
  const auto compute_time
      = std::chrono::duration_cast<std::chrono::nanoseconds>(compute_end_time - compute_start_time)
            .count();

  // Report results
  fmt::println("Computed value of pi = {}", total_area);
  fmt::println("Error from actual value of pi = {}", fabs(total_area - std::numbers::pi));
  fmt::println("Time elapsed computing pi: {} ns", compute_time);

  return EXIT_SUCCESS;
}