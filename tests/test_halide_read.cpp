
#include <libimread/libimread.hpp>
#include <libimread/halide.hh>

using namespace Halide;
using U8Image = Image<uint8_t>;

namespace ext {
    #include <libimread/private/image_io.h>
}

#include "include/catch.hpp"

TEST_CASE("Read a JPEG and rewrite it as a PNG via image_io.h", "[read-jpeg-write-png]") {
    U8Image halim = im::halide::read(
        "/Users/fish/Downloads/480165102_76fbb0739a_o.jpg");
    ext::save(halim,
        "/tmp/YO_DOGG222.png");
    // U8Image halim2 = im::halide::read(
    //     "/tmp/YO_DOGG.png");
}

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
    
    WARN( "1.extent(0) = " << halim.extent(0) );
    WARN( "1.extent(1) = " << halim.extent(1) );
    WARN( "1.extent(2) = " << halim.extent(2) );
    
    WARN( "1.stride(0) = " << halim.stride(0) );
    WARN( "1.stride(1) = " << halim.stride(1) );
    WARN( "1.stride(2) = " << halim.stride(2) );
    
    U8Image halim2 = im::halide::read(
        "/Users/fish/Downloads/480165102_76fbb0739a_o.jpg");
    
    WARN( "2.extent(0) = " << halim2.extent(0) );
    WARN( "2.extent(1) = " << halim2.extent(1) );
    WARN( "2.extent(2) = " << halim2.extent(2) );
    
    WARN( "2.stride(0) = " << halim2.stride(0) );
    WARN( "2.stride(1) = " << halim2.stride(1) );
    WARN( "2.stride(2) = " << halim2.stride(2) );
    
}
