/**
 * This program is a minimal MPI example.
 */

#include <fmt/base.h>
#include <mpi.h>

auto main(int argc, char **argv) -> int {
  // All MPI calls need to happen between MPI_Init() and MPI_Finalize()
  MPI_Init(&argc, &argv);

  // Get the total
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of this process
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Print off a hello world message
  fmt::println("Hello from rank {} / {}", world_rank, world_size);

  // Finalize the MPI environment.
  MPI_Finalize();
}
