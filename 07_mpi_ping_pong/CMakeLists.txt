cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# -----------------------------------------
# Project
# -----------------------------------------

project(
  mpi_ping_pong
  VERSION 1.0.0
  LANGUAGES CXX)

# -----------------------------------------
# Target sources
# -----------------------------------------

set(SOURCE_LIST "${PROJECT_SOURCE_DIR}/src/main.cpp")

# -----------------------------------------
# Executable target
# -----------------------------------------

add_executable(mpi_ping_pong ${SOURCE_LIST})
target_compile_features(mpi_ping_pong PUBLIC cxx_std_20)
set_target_properties(mpi_ping_pong PROPERTIES OUTPUT_NAME "mpi_ping_pong")

# -----------------------------------------
# Compilers flags and options
# -----------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(
    mpi_ping_pong
    PUBLIC -Og
           -g3
           -ggdb3
           -fno-omit-frame-pointer
           -Wall
           -Wextra
           -Wpedantic
           -Walloca
           -Wcast-qual
           -Wformat=2
           -Wformat-security
           -Wnull-dereference
           -fstack-protector
           -Wvla
           -Wconversion
           -Warray-bounds
           -Wuninitialized
           -Wimplicit-fallthrough
           -Wpointer-arith
           -Wfloat-equal
           -Wswitch-enum
           -Wno-switch-enum)

  target_link_options(mpi_ping_pong PUBLIC -Og -g3 -ggdb3)
  target_link_libraries(mpi_ping_pong PUBLIC debuginfod)
  target_link_libraries(mpi_ping_pong PUBLIC unwind)
endif()

# -----------------------------------------
# Link and build order dependencies
# -----------------------------------------

target_link_libraries(mpi_ping_pong PRIVATE fmt::fmt MPI::MPI_CXX)
