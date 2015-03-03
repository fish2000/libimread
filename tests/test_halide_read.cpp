
//#include <stdint.h>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>

#include "include/catch.hpp"

using U8Image = Halide::Image<uint8_t>;

TEST_CASE("Read a JPEG", "[read-jpeg]") {
    U8Image halim = im::halide::read(
        "/Users/fish/Downloads/480165102_76fbb0739a_o.jpg");
    
    CHECK( halim.data() != nullptr );
    CHECK( halim.data() != 0 );
}

TEST_CASE("Check the dimensions of a new image", "[image-dims]") {
    U8Image halim = im::halide::read(
        "/Users/fish/Downloads/dd1c09d792a053508ef7a785dc28cbc9.jpg");
    
    //WARN( "extent[0] = " << buf->extent[0] );
    
    // CAPTURE( halim.extent(0) );
    // CAPTURE( halim.extent(1) );
    // CAPTURE( halim.extent(2) );
    // //CAPTURE( halim.extent(3) );
    //
    // CAPTURE( halim.stride(0) );
    // CAPTURE( halim.stride(1) );
    // CAPTURE( halim.stride(2) );
    // //CAPTURE( halim.stride(3) );
    
    WARN( "extent(0) = " << halim.extent(0) );
    WARN( "extent(1) = " << halim.extent(1) );
    WARN( "extent(2) = " << halim.extent(2) );
    
    WARN( "stride(0) = " << halim.stride(0) );
    WARN( "stride(1) = " << halim.stride(1) );
    WARN( "stride(2) = " << halim.stride(2) );
}
