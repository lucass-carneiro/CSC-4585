/*
 * This is Conway's game of life parallelized using MPI
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <experimental/mdspan>
#include <fmt/format.h>
#include <mpi.h>
#include <random>
#include <toml++/toml.hpp>
#include <vector>

namespace stde = std::experimental;

using usize = std::size_t;
using u8 = std::uint8_t;

// Store simulation data
enum IDType : int { glider_id, random_id };

struct SimulationData {
  usize grid_size{32};       // Gobal grid size. The grid is always square.
  usize generations{32};     // Numbner of generations
  usize stats_every{1};      // Output statistics every STATS_EVERY iterations
  usize data_every{1};       // Dump data to disk every DATA_EVERY iterations
  usize random_seed{64};     // Random seed used in initialization
  IDType id_type{random_id}; // Type of initial data
};

// Compute local stripe partitioning (rows per rank)
struct Partition {
  int rank{0};         // Rank that owns the partition
  int size{0};         // Total number of ranks
  usize local_rows{0}; // Number of data rows (excluding halo rows)
  usize row_offset{0}; // Global index of the first row owned by this rank.
};

Partition compute_partition(const SimulationData &sd, int rank, int size) {
  /*
   * To allow for grid_size be divisible by size, we will use the same trick we used in the first
   * OpenMP parallelization example and distribuite the cell remainder across ranks allow
   * sd.grid_size not
   */
  const auto base = sd.grid_size / static_cast<usize>(size);
  const auto rem = sd.grid_size % static_cast<usize>(size);

  const auto local = base + (static_cast<usize>(rank) < rem ? 1 : 0);
  const auto offset = base * static_cast<usize>(rank) + std::min(static_cast<usize>(rank), rem);

  return Partition{rank, size, local, offset};
}

// Print only on rank zero
#define root_println(format, ...)                                                                  \
  if (rank == 0) {                                                                                 \
    fmt::println(format __VA_OPT__(, ) __VA_ARGS__);                                               \
  }

// Get a pointer to the start of a row. MPI needs this
static inline auto row_ptr(const SimulationData &sd, u8 *data_ptr, usize r) -> u8 * {
  return data_ptr + (r * sd.grid_size);
};

auto parse_sim_data(const char *file_path) -> SimulationData {
  SimulationData data;

  const auto toml_file = toml::parse_file(file_path);

  data.grid_size = static_cast<usize>(toml_file["general"]["grid_size"].value_or(32));
  data.generations = static_cast<usize>(toml_file["general"]["generations"].value_or(32));
  data.stats_every = static_cast<usize>(toml_file["general"]["stats_every"].value_or(1));
  data.data_every = static_cast<usize>(toml_file["general"]["data_every"].value_or(1));
  data.random_seed = static_cast<usize>(toml_file["id"]["random_seed"].value_or(64));

  const auto id_type = toml_file["id"]["id_type"].value_or("random");

  if (strcmp(id_type, "random") == 0) {
    data.id_type = IDType::random_id;
  } else if (strcmp(id_type, "glider") == 0) {
    data.id_type = IDType::glider_id;
  }

  return data;
}

int main(int argc, char **argv) {
  using std::swap;

  MPI_Init(&argc, &argv);

  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 2) {
    root_println("Usage: {} <config-file.toml>", argv[0]);
    return EXIT_FAILURE;
  }

  const auto sd = parse_sim_data(argv[1]);

  if (static_cast<usize>(size) > sd.grid_size) {
    root_println("Warning: more MPI ranks ({}) than rows in grid ({}). Behavior will still be "
                 "periodic but some ranks will get zero rows.",
                 size, sd.grid_size);
  }

  const auto p = compute_partition(sd, rank, size);

  /*
   * This rank has no data rows but we still must participate in communications. For simplicity,
   * we will terminate these ranks and continue on with the ones that do have some data to work on.
   */
  if (p.local_rows == 0) {
    fmt::println(
        "Rank {} got 0 rows due to grid size ({}) < num. procs ({}). Exiting those ranks.\n", rank,
        sd.grid_size, size);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  /*
   * Buffers: we allocate (local_rows + 2) rows to hold top and bottom halos.
   * Layout:
   *  row 0 => top halo (from neighbor above)
   *  rows 1..local_rows => actual data, row
   *  local_rows + 1 => bottom halo
   */
  const auto rows_with_halo = p.local_rows + 2;
  std::vector<u8> grid_buf(rows_with_halo * sd.grid_size);
  std::vector<u8> next_buf(rows_with_halo * sd.grid_size);

  /*
   * An mdspan is a multi dimensional view of a contiguous block of data. Being a view, it does not
   * own the data, it only allows us to interact with it on a different way. This is similar to
   * reshaping numpy arrays, if you used those before
   */
  stde::mdspan grid(grid_buf.data(), rows_with_halo, sd.grid_size);
  stde::mdspan next_grid(next_buf.data(), rows_with_halo, sd.grid_size);

  // Initialize the grid
  switch (sd.id_type) {
  case random_id: {
    std::mt19937_64 rng(sd.random_seed + static_cast<usize>(rank));
    std::uniform_int_distribution<uint8_t> bit(0, 1);

    for (usize r = 1; r <= p.local_rows; r++) {
      for (usize c = 0; c < sd.grid_size; c++) {
        grid(r, c) = bit(rng);
      }
    }

    break;
  }

  case glider_id:
    grid(1, 0) = 0;
    grid(1, 1) = 1;
    grid(1, 2) = 0;

    grid(2, 0) = 0;
    grid(2, 1) = 0;
    grid(2, 2) = 1;

    grid(3, 0) = 1;
    grid(3, 1) = 1;
    grid(3, 2) = 1;

    break;
  }

  // Get the ranks of up and down neighbours
  const int up = (rank - 1 + size) % size;
  const int down = (rank + 1) % size;

  // Loop over generations
  for (usize step = 0; step < sd.generations; step++) {
    /*
     * Post non-blocking receives for halos:
     * Receive top halo (row 0) from neighbor 'up' (they will send their bottom data row)
     * Receive bottom halo (row local_rows + 1) from neighbor 'down' (they will send their top data
     * row).
     */
    MPI_Request reqs[4];
    MPI_Irecv(row_ptr(sd, grid_buf.data(), 0), static_cast<int>(sd.grid_size), MPI_UNSIGNED_CHAR,
              up, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(row_ptr(sd, grid_buf.data(), p.local_rows + 1), static_cast<int>(sd.grid_size),
              MPI_UNSIGNED_CHAR, down, 1, MPI_COMM_WORLD, &reqs[1]);

    /*
     * Post non-blocking sends for the rows we have and our neighbours will need.
     * Send our bottom data row (row p.local_rows) to 'down' with tag 0 (so that down receives into
     * its top halo)
     * Send our top real row (row 1) to 'up' with tag 1 (so that up receives into its bottom halo)
     */
    MPI_Isend(row_ptr(sd, grid_buf.data(), p.local_rows), static_cast<int>(sd.grid_size),
              MPI_UNSIGNED_CHAR, down, 0, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(row_ptr(sd, grid_buf.data(), 1), static_cast<int>(sd.grid_size), MPI_UNSIGNED_CHAR,
              up, 1, MPI_COMM_WORLD, &reqs[3]);

    /*
     * Wait for all four operations to complete before computing
     * Note that we ignore the status of the communications and don't check for possible errors.
     * What could go wrong after all? :)
     *
     * Is there anything we could do to improve this design?
     */
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    /*
     * We have all the data we need. We can now compute the next generation in the game.
     * Remember that we update only the local data (the non halo cells) and use the halo cells when
     * necessary.
     */
    for (usize r = 1; r <= p.local_rows; r++) {
      for (usize c = 0; c < sd.grid_size; c++) {
        // Periodic row boundary condition
        int left = (c == 0) ? static_cast<int>(sd.grid_size - 1) : static_cast<int>(c - 1);
        int right = (c + 1 == sd.grid_size) ? 0 : static_cast<int>(c + 1);

        int nsum = 0;
        // three rows: r-1, r, r+1
        nsum += grid(r - 1, left);
        nsum += grid(r - 1, c);
        nsum += grid(r - 1, right);

        nsum += grid(r, left);
        // skip grid(r,c) itself
        nsum += grid(r, right);

        nsum += grid(r + 1, left);
        nsum += grid(r + 1, c);
        nsum += grid(r + 1, right);

        u8 cur = grid(r, c);
        u8 nxt = 0;

        if (cur == 1) {
          // live cell: survives with 2 or 3 neighbors
          nxt = (nsum == 2 || nsum == 3) ? 1 : 0;
        } else {
          // dead cell: becomes live if exactly 3 neighbors
          nxt = (nsum == 3) ? 1 : 0;
        }

        next_grid(r, c) = nxt;
      }
    }

    // Diagnostics
    if (step % sd.stats_every == 0) {
      long local_sum = 0;
      for (usize r = 1; r <= p.local_rows; ++r) {
        for (usize c = 0; c < sd.grid_size; ++c) {
          local_sum += grid(r, c);
        }
      }

      long global_sum = 0;
      MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

      root_println("Iteration {}. Live cells {}", step, global_sum);
    }

    /*
     * Save data to disk. All processes dump their local portions of the grid but we save the file
     * with coordinates relative to the  global grid. This makes it easier for us to plot the state.
     */
    if (step % sd.data_every == 0) {
      auto out_file = fopen(fmt::format("gol_it_{:08}_rank_{:08}.dat", step, rank).c_str(), "w");

      fmt::println(out_file, "#1:row    2:col    3:state");

      for (std::size_t r = 1; r <= p.local_rows; ++r) {
        for (std::size_t c = 0; c < sd.grid_size; ++c) {
          const auto global_r = p.row_offset + (r - 1);
          fmt::println(out_file, "{}    {}    {}", global_r, c, grid(r, c));
        }
      }

      fclose(out_file);
    }

    /*
     * Swap the scratch buffer with the current state buffer
     * Note that we are alswo swapping the halos. That does not matter, as they get written with the
     * correct data on every iteration.
     */
    std::swap(grid_buf, next_buf);

    // We swapped buffer pointers, so let's not forget to update our views!
    grid = stde::mdspan(grid_buf.data(), rows_with_halo, sd.grid_size);
    next_grid = stde::mdspan(next_buf.data(), rows_with_halo, sd.grid_size);
  }

  MPI_Finalize();
  return 0;
}