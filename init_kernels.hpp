#ifndef INIT_KERNELS_HPP
#define INIT_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include "parameters.hpp"


using grid_type = Kokkos::View<double*>;

struct DirichletInit
{
  grid_type g_old;
  grid_type g_new;
  int start_owned_global;
  int halo_top;

  DirichletInit( grid_type g_old_, grid_type g_new_, int start_owned_global_, int halo_top_ ):
    g_old( g_old_ ), g_new( g_new_ ),
    start_owned_global( start_owned_global_ ), halo_top( halo_top_ ) {}

  KOKKOS_INLINE_FUNCTION
  void operator()( const int idx ) const
  {
    const int local_row  = idx / width;
    const int global_row = start_owned_global + (local_row - halo_top);

    double val = 0.0;
    if( global_row == 0 || global_row == (height - 1) )
    {
      val = 40.0;
    }

    g_old( idx ) = val;
    g_new( idx ) = val;
  }
};


struct UniformInit
{
  grid_type g_old;
  grid_type g_new;
  double init_val;

  UniformInit( grid_type g_old_, grid_type g_new_, double val ):
    g_old( g_old_ ), g_new( g_new_ ), init_val( val ) {}

  KOKKOS_INLINE_FUNCTION
  void operator()( const int idx ) const
  {
    g_old( idx ) = init_val;
    g_new( idx ) = init_val;
  }
};


#endif /* INIT_KERNELS_HPP */
