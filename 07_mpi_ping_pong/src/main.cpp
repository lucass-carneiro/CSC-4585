#include <fmt/core.h>
#include <mpi.h>
#include <string>

auto main(int argc, char **argv) -> int {
  MPI_Init(&argc, &argv);

  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size < 2) {
    if (world_rank == 0) {
      fmt::println("World size must be at least 2 for ping pong");
    }
    MPI_Finalize();
    return 1;
  }

  constexpr int PING_PONG_LIMIT = 5;
  int ping_pong_count = 0;
  int partner_rank = (world_rank + 1) % 2;

  while (ping_pong_count < PING_PONG_LIMIT) {
    if (world_rank == ping_pong_count % 2) {
      const auto msg = fmt::format("Ping {} from {}", ping_pong_count, world_rank);

      // Send message length first
      auto msg_size = msg.size();
      MPI_Send(&msg_size, 1, MPI_UNSIGNED_LONG, partner_rank, 0, MPI_COMM_WORLD);

      // Send the actual characters
      MPI_Send(msg.c_str(), static_cast<int>(msg_size), MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD);

      fmt::println("{} sent \"{}\"", world_rank, msg);

      ping_pong_count++;
    } else {
      // Receive message length
      unsigned long int msg_size = 0;
      MPI_Recv(&msg_size, 1, MPI_UNSIGNED_LONG, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Receive actual message
      std::string msg(msg_size, '\0');
      MPI_Recv(msg.data(), static_cast<int>(msg_size), MPI_CHAR, partner_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      fmt::println("{} received \"{}\"", world_rank, msg);
      ping_pong_count++;
    }
  }

  MPI_Finalize();
  return 0;
}
