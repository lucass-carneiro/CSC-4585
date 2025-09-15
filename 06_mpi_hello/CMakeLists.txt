cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# -----------------------------------------
# Project
# -----------------------------------------

project(
  mpi_hello
  VERSION 1.0.0
  LANGUAGES CXX)

# -----------------------------------------
# Target sources
# -----------------------------------------

set(SOURCE_LIST "${PROJECT_SOURCE_DIR}/src/main.cpp")

# -----------------------------------------
# Executable target
# -----------------------------------------

add_executable(mpi_hello ${SOURCE_LIST})
target_compile_features(mpi_hello PUBLIC cxx_std_20)
set_target_properties(mpi_hello PROPERTIES OUTPUT_NAME "mpi_hello")

# -----------------------------------------
# Compilers flags and options
# -----------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(
    mpi_hello
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

  target_link_options(mpi_hello PUBLIC -Og -g3 -ggdb3)
  target_link_libraries(mpi_hello PUBLIC debuginfod)
  target_link_libraries(mpi_hello PUBLIC unwind)
endif()

# -----------------------------------------
# Link and build order dependencies
# -----------------------------------------

target_link_libraries(mpi_hello PRIVATE fmt::fmt MPI::MPI_CXX)
