#include <cstdio>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "parameters.hpp"


int main( int argc, char *argv[] )
{
  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );
  {
    if constexpr( height % mpi_procs != 0 )
    {
      printf("Can't split horizontally: height=%d is not divisible by mpi_procs=%d\n", height, mpi_procs);
      exit(1);
    }

    int mpi_rank;
    int mpi_count;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &mpi_count );
    if( mpi_count != mpi_procs )
    {
      printf("mpi_procs=%d differs from the number of instantiated MPI processes (%d)\n", mpi_procs, mpi_count);
      exit(1);
    }

    using halo_view = Kokkos::View<double*, Kokkos::HostSpace>;
    MPI_Barrier( MPI_COMM_WORLD );

    int local_height = (height / mpi_procs) + 1;
    if( mpi_rank != 0 && mpi_rank != mpi_count-1 ) local_height++;
    int local_size = width * local_height;

    Kokkos::View<double*> d_old( "old", local_size );
    Kokkos::View<double*> d_new( "new", local_size );

    // this is just an example init kernel, you can change it
    // if you wish
    const int chunk = height / mpi_procs;
    const int start_owned_global = mpi_rank * chunk;
    const int halo_top = (mpi_rank > 0) ? 1 : 0;

    Kokkos::parallel_for( "init_kernel", local_size, KOKKOS_LAMBDA( const int idx ){
      const int local_row  = idx / width;
      const int global_row = start_owned_global + (local_row - halo_top);

      double val = 0.0;
      if( global_row == 0 || global_row == (height - 1) )
      {
        val = 40.0;
      }

      d_old( idx ) = val;
      d_new( idx ) = val;
    });

    for( int it = 0; it < max_it; it++)
    {

      Kokkos::parallel_for(
        "transition_kernel", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {local_height, width}),
        KOKKOS_LAMBDA( const int i, const int j ){
          // Those are set like this because I am choosing to not
          // update the top and bottom rows of the ENTIRE grid
          int first_row = 1;
          int last_row  = local_height-2;

          if( i >= first_row && i <= last_row )
          {
            double up    = d_old( (i-1) * width + j );
            double down  = d_old( (i+1) * width + j );

            int j_left   = ( j > 0 )? (j-1) : j;
            int j_right  = ( j < width-1 )? (j+1) : j;
            double left  = d_old( i * width + j_left );
            double right = d_old( i * width + j_right );

            double curr = d_old( i * width + j );
            d_new( i * width + j ) = curr + 0.25 * (up + down + left + right - 4.0 * curr);
          }

      });
      Kokkos::fence();
     
      if( mpi_count > 1 )
      {
        MPI_Request reqs[4];
        int nreq = 0;

        size_t send_down_offset = width * (local_height - 2);
        size_t recv_down_offset = width * (local_height - 1);
        size_t send_up_offset   = width;
        size_t recv_up_offset   = 0;

        halo_view recv_up_buf, recv_down_buf, send_up_buf, send_down_buf;

        auto recv_up_sv   = Kokkos::subview( d_new, std::make_pair(recv_up_offset, recv_up_offset + width) );
        auto recv_down_sv = Kokkos::subview( d_new, std::make_pair(recv_down_offset, recv_down_offset + width) );
        auto send_up_sv   = Kokkos::subview( d_new, std::make_pair(send_up_offset, send_up_offset + width) );
        auto send_down_sv = Kokkos::subview( d_new, std::make_pair(send_down_offset, send_down_offset + width) );

        if( mpi_rank > 0 )
        {
          recv_up_buf = Kokkos::create_mirror_view( Kokkos::HostSpace(), recv_up_sv );
          send_up_buf = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), send_up_sv );
          
          MPI_Irecv( recv_up_buf.data(), width, MPI_DOUBLE, mpi_rank-1, 666, MPI_COMM_WORLD, &reqs[nreq++] );
          MPI_Isend( send_up_buf.data(), width, MPI_DOUBLE, mpi_rank-1, 666, MPI_COMM_WORLD, &reqs[nreq++] );
        }
        if( mpi_rank < mpi_count - 1 )
        {
          recv_down_buf = Kokkos::create_mirror_view( Kokkos::HostSpace(), recv_down_sv );
          send_down_buf = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), send_down_sv );
          
          MPI_Irecv( recv_down_buf.data(), width, MPI_DOUBLE, mpi_rank+1, 666, MPI_COMM_WORLD, &reqs[nreq++] );
          MPI_Isend( send_down_buf.data(), width, MPI_DOUBLE, mpi_rank+1, 666, MPI_COMM_WORLD, &reqs[nreq++] );
        }

        MPI_Waitall( nreq, reqs, MPI_STATUSES_IGNORE );

        if( mpi_rank > 0 )
        {
          Kokkos::deep_copy( recv_up_sv, recv_up_buf );
        }
        if( mpi_rank < mpi_count - 1 )
        {
          Kokkos::deep_copy( recv_down_sv, recv_down_buf );
        }
      }

      std::swap( d_old, d_new );

    }

    double min_temp_local = *Kokkos::Experimental::min_element( Kokkos::DefaultExecutionSpace(), d_old );
    double max_temp_local = *Kokkos::Experimental::max_element( Kokkos::DefaultExecutionSpace(), d_old );

    double min_temp_global = 0.0;
    double max_temp_global = 0.0;
    MPI_Allreduce( &min_temp_local, &min_temp_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
    MPI_Allreduce( &max_temp_local, &max_temp_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );

    Kokkos::View<unsigned char*> rgb_buf( "rgb_buffer", local_size*3 );
    Kokkos::parallel_for( "write_rgb_buffer_kernel", local_size, KOKKOS_LAMBDA( const int idx ){

      double t = (d_old( idx ) - min_temp_global) / (max_temp_global - min_temp_global);
      if (t < 0.0) t = 0.0;
      if (t > 1.0) t = 1.0;

      double r = 0.0, g = 0.0, b = 0.0;

      if (t < 0.25)
      {
          double f = t / 0.25;
          r = 0.0;
          g = 0.0;
          b = 30.0 + f * (255.0 - 30.0);
      } 
      else if (t < 0.50)
      {
          double f = (t - 0.25) / 0.25;
          r = 0.0;
          g = f * 255.0;
          b = 255.0;
      } 
      else if (t < 0.75)
      {
          double f = (t - 0.50) / 0.25;
          r = f * 255.0;
          g = 255.0 - f * (255.0 - 140.0);
          b = 255.0 * (1.0 - f);
      } 
      else
      {
          double f = (t - 0.75) / 0.25;
          r = 255.0;
          g = 140.0 * (1.0 - f);
          b = 0.0;
      }

      rgb_buf( idx * 3 + 0 ) = static_cast<unsigned char>(r);
      rgb_buf( idx * 3 + 1 ) = static_cast<unsigned char>(g);
      rgb_buf( idx * 3 + 2 ) = static_cast<unsigned char>(b);
    });

    auto h_rgb = Kokkos::create_mirror_view_and_copy( rgb_buf );

    if( mpi_count > 1 )
    {
      const int elem_count = 3 * (height / mpi_count) * width;
      if( mpi_rank == 0 )
      {
        std::ofstream out( OUTPUT_PPM, std::ios::binary );
        const std::string header = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
        out.write( header.c_str(), header.length() );
        out.write( reinterpret_cast<const char*>(h_rgb.data()), elem_count );

        std::vector<unsigned char> recv_buf( elem_count );
        for( int i = 1; i < mpi_count; ++i )
        {
          MPI_Recv( recv_buf.data(), elem_count, MPI_UNSIGNED_CHAR, i, 666, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
          out.write( reinterpret_cast<const char*>(recv_buf.data()), elem_count );
        }
      }
      else
      {
        MPI_Request req;
        MPI_Isend( h_rgb.data() + 3*width, elem_count, MPI_UNSIGNED_CHAR, 0, 666, MPI_COMM_WORLD, &req );
        MPI_Wait( &req, MPI_STATUS_IGNORE );
      }
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
