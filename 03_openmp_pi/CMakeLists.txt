cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# -----------------------------------------
# Project
# -----------------------------------------

project(
  openmp_pi
  VERSION 1.0.0
  LANGUAGES CXX
)

# -----------------------------------------
#  Target sources
# -----------------------------------------

set(
  SOURCE_LIST
  "${PROJECT_SOURCE_DIR}/src/main.cpp"
)

# -----------------------------------------
# Executable target
# -----------------------------------------

add_executable(openmp_pi ${SOURCE_LIST})
target_compile_features(openmp_pi PUBLIC cxx_std_20)
set_target_properties(openmp_pi PROPERTIES OUTPUT_NAME "openmp_pi")

# -----------------------------------------
# Compilers flags and options
# -----------------------------------------

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(
    openmp_pi
    PUBLIC
    -Og
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
    -Wno-switch-enum
  )

  target_link_options(openmp_pi PUBLIC -Og -g3 -ggdb3)
  target_link_libraries(openmp_pi PUBLIC debuginfod)
  target_link_libraries(openmp_pi PUBLIC unwind)
endif ()

# -----------------------------------------
# Link and build order dependencies
# -----------------------------------------

target_link_libraries(
  openmp_pi
  PRIVATE
  fmt::fmt
  OpenMP::OpenMP_CXX
)