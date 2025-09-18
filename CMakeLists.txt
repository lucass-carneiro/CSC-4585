cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# -----------------------------------------
# Project
# -----------------------------------------

project(
  csc-4585
  VERSION 1.0.0
  LANGUAGES CXX)

# Generate compile_commands.json for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

# -----------------------------------------
# In-source build guard
# -----------------------------------------

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# -----------------------------------------
# External dependencies
# -----------------------------------------

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(tl-expected CONFIG REQUIRED)
find_package(mdspan CONFIG REQUIRED)
find_package(tomlplusplus CONFIG REQUIRED)

# -----------------------------------------
# Targets
# -----------------------------------------

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/00_openmp_hello)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/01_openmp_non_det)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/02_serial_pi)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/03_openmp_pi)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/04_openmp_pi_critical)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/05_openmp_pi_parallel_for)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/06_mpi_hello)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/07_mpi_ping_pong)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/08_mpi_gol)
