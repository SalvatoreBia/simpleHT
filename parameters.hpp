#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <string>

#define OUTPUT_PPM "output.ppm"

constexpr size_t width    = 240;
constexpr size_t height   = 240;
constexpr size_t size     = width * height;

constexpr int mpi_procs   = 4;

constexpr int max_it      = 10000;

constexpr bool periodic_x = true;

#endif /* PARAMETERS_HPP */
