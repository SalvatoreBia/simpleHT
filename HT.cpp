#include <cstdio>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <fstream>
#include <iostream>

#define OUTPUT_PPM "output.ppm"


int main( int argc, char *argv[] )
{
  Kokkos::initialize( argc, argv );
  {

    constexpr int width  = 100;
    constexpr int height = 100;
    constexpr int size   = width * height;

    Kokkos::View<double*> d_old( "old", size );
    Kokkos::View<double*> d_new( "new", size );


    Kokkos::parallel_for( "init_kernel", size, KOKKOS_LAMBDA( const int idx ){
      d_old( idx ) = 0.0;
      d_new( idx ) = 0.0;
      
      if( idx < width || idx >= (size - width) )
      {
        d_old( idx ) = 40.0;
        d_new( idx ) = 40.0;
      }
    });

    constexpr int max_it = 100;
    for( int it = 0; it < max_it; it++)
    {

      Kokkos::parallel_for(
        "transition_kernel", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {width, height}),
        KOKKOS_LAMBDA( const int i, const int j ){
       
          if( i > 0 && i < height - 1 )
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

      std::swap( d_old, d_new );

    }

    double min_temp = *Kokkos::Experimental::min_element( Kokkos::DefaultExecutionSpace(), d_new );
    double max_temp = *Kokkos::Experimental::max_element( Kokkos::DefaultExecutionSpace(), d_new );

    Kokkos::View<unsigned char*> rgb_buf( "rgb_buffer", size*3 );
    Kokkos::parallel_for( "write_rgb_buffer_kernel", size, KOKKOS_LAMBDA( const int idx ){

      double t = (d_new( idx ) - min_temp) / (max_temp - min_temp);
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

    std::ofstream out( OUTPUT_PPM, std::ios::binary );
    const std::string header = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
    out.write( header.c_str(), header.length() );
    out.write( reinterpret_cast<const char*>(h_rgb.data()), size * 3 );
  }
  Kokkos::finalize();
  return 0;
}
