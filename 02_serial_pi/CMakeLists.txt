cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# -----------------------------------------
# Project
# -----------------------------------------

project(
  serial_pi
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

add_executable(serial_pi ${SOURCE_LIST})
target_compile_features(serial_pi PUBLIC cxx_std_20)
set_target_properties(serial_pi PROPERTIES OUTPUT_NAME "serial_pi")

# -----------------------------------------
# Compilers flags and options
# -----------------------------------------

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(
    serial_pi
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

  target_link_options(serial_pi PUBLIC -Og -g3 -ggdb3)
  target_link_libraries(serial_pi PUBLIC debuginfod)
  target_link_libraries(serial_pi PUBLIC unwind)
endif ()

# -----------------------------------------
# Link and build order dependencies
# -----------------------------------------

target_link_libraries(
  serial_pi
  PRIVATE
  fmt::fmt
)