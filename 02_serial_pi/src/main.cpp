/**
 * This program computes the value of pi by integrating the function
 * $$ \int_{0}^{1} \frac{4}{\sqrt{1 + x^2}} dx $$
 *
 * To do that, we partition the interval [0, 1] into n parallelograms that approximate the function
 * and sum their areas serially.
 *
 * We compare the final result with C++20's std::numbers::pi
 *
 * W
 */
#include <argparse/argparse.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fmt/base.h>
#include <numbers>

auto integrand(double x) -> double { return 4.0 / (1.0 + x * x); }

auto main(int argc, char **argv) -> int {
  // Argument handling
  argparse::ArgumentParser program("serial_pi");

  constexpr auto num_blocks_arg_str = "num_blocks";
  using num_blocks_t = std::uint64_t;

  program.add_argument(num_blocks_arg_str)
      .help("Number of blocks to use for the integration")
      .required()
      .scan<'u', num_blocks_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    fmt::println("CLI error: {}", err.what());
    return EXIT_FAILURE;
  }

  const auto num_blocks = program.get<num_blocks_t>(num_blocks_arg_str);

  // Partitioning the interval
  fmt::println("Computing pi using {} blocks", num_blocks);

  const double interval_start = 0.0;
  const double interval_end = 1.0;
  const auto interval_step = (interval_end - interval_start) / static_cast<double>(num_blocks);

  // Loop over blocks and compute areas
  double total_area = 0.0;

  const auto compute_start_time = std::chrono::steady_clock::now();

  for (num_blocks_t i = 0; i < num_blocks; i++) {
    const auto x0 = interval_start + static_cast<double>(i) * interval_step;
    const auto x1 = x0 + interval_step;

    const auto y0 = integrand(x0);
    const auto y1 = integrand(x1);

    const auto tallest{y0 > y1 ? y0 : y1};
    const auto shortest{y0 < y1 ? y0 : y1};

    const auto rect_area = interval_step * shortest;
    const auto tri_area = interval_step * (tallest - shortest) / 2.0;

    total_area += rect_area + tri_area;
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