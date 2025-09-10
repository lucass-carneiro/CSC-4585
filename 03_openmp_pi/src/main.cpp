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
#include <cstdio>
#include <cstdlib>
#include <fmt/base.h>
#include <numbers>
#include <omp.h>
#include <tuple>
#include <vector>

using num_blocks_t = std::uint64_t;
using num_threads_t = int;

auto integrand(double x) -> double { return 4.0 / (1.0 + x * x); }

template <bool verbose> static auto compute_pi(num_blocks_t num_blocks, num_threads_t num_threads) {
  using std::min;

  // Partitioning the interval
  if constexpr (verbose) {
    fmt::println("Computing pi using {} blocks", num_blocks);
  }

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

    if constexpr (verbose) {
      if (thread_id == 0) {
        fmt::println("Requested / available threads: {} / {}", num_threads, actual_num_threads);
      }
    }

    const auto blocks_per_thread = num_blocks / actual_num_threads;
    const auto remainder = num_blocks % actual_num_threads;

    const auto my_blocks = blocks_per_thread + (thread_id < remainder ? 1 : 0);
    const auto start_block = thread_id * blocks_per_thread + min(thread_id, remainder);

    if constexpr (verbose) {
      fmt::println("Thread {} is working on {} blocks, starting on block {} and ending on block {}",
                   thread_id, my_blocks, start_block, start_block + my_blocks);
    }

    double thread_area = 0;

    for (std::uint64_t i = 0; i < my_blocks; i++) {
      const auto x0 = static_cast<double>(start_block + i) * interval_step;
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

  return std::make_tuple(total_area, compute_time);
}

auto main(int argc, char **argv) -> int {
  using std::fclose;
  using std::fopen;

  // Argument handling
  argparse::ArgumentParser program("openmp_pi");

  constexpr auto num_blocks_arg_str = "num_blocks";
  program.add_argument(num_blocks_arg_str)
      .help("Number of blocks to use for the integration")
      .required()
      .scan<'u', num_blocks_t>();

  constexpr auto num_threads_arg_string = "num_threads";
  program.add_argument(num_threads_arg_string)
      .help("Number of threads to use when integrating")
      .required()
      .scan<'i', num_threads_t>();

  constexpr auto scaling_test_arg_string = "--scaling";
  program.add_argument(scaling_test_arg_string)
      .help("Colect metrics for a scaling test")
      .default_value(false)
      .implicit_value(true);

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    fmt::println("CLI error: {}", err.what());
    return EXIT_FAILURE;
  }

  const auto num_blocks = program.get<num_blocks_t>(num_blocks_arg_str);
  const auto num_threads = program.get<num_threads_t>(num_threads_arg_string);
  const auto do_scaling_test = program.get<bool>(scaling_test_arg_string);

  // Standard run
  const auto [computed_pi, compute_time] = compute_pi<true>(num_blocks, num_threads);

  fmt::println("Computed value of pi = {}", computed_pi);
  fmt::println("Error from actual value of pi = {}", fabs(computed_pi - std::numbers::pi));
  fmt::println("Time elapsed computing pi: {} ns", compute_time);

  // Statistics run
  if (do_scaling_test) {
    fmt::println("Doing scaling testing ...");

    auto out_file = fopen("openmp_pi_scaling.dat", "w");
    fmt::println(out_file, "#1: Threads    2: Time (ns)    3: Speedup");

    constexpr int repeat = 10;

    double first_time_avg = 0.0;

    for (int i = 1; i <= num_threads; i++) {

      long time_sum = 0;

      for (int j = 0; j < repeat; j++) {
        const auto [_, time] = compute_pi<false>(num_blocks, i);
        time_sum += time;
      }

      const auto time_avg = static_cast<double>(time_sum) / static_cast<double>(repeat);

      if (i == 1) {
        first_time_avg = time_avg;
      }

      const auto speedup = first_time_avg / time_avg;

      fmt::println(out_file, "{}    {:.16e}    {:.16e}", i, time_avg, speedup);
    }

    fclose(out_file);
  }

  return EXIT_SUCCESS;
}