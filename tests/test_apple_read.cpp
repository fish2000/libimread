
#include <libimread/libimread.hpp>
#include <libimread/halide.hh>

#include "include/catch.hpp"

namespace {

    using namespace Halide;
    using U8Image = Image<uint8_t>;

    namespace ext {
        #include <libimread/private/image_io.h>
    }

    TEST_CASE("[apple] Read a JPEG and rewrite it as a PNG via image_io.h", "[apple-read-jpeg-write-png]") {
        U8Image halim = im::apple::read(
            "../tests/data/tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg");
        ext::save(halim, "/tmp/apple_YO_DOGG222.png");
    }

    TEST_CASE("[apple] Read a PNG", "[apple-read-png]") {
        U8Image halim = im::apple::read(
            "../tests/data/roses_512_rrt_srgb.png");
        ext::save(halim, "/tmp/apple_OH_DAWG666.png");
    }

    TEST_CASE("[apple] Read a TIFF", "[apple-read-tiff]") {
        U8Image halim = im::apple::read(
            "../tests/data/ptlobos.tif");
        ext::save(halim, "/tmp/apple_TIFF_DUG986.png");
    }

    TEST_CASE("[apple] Read a JPEG", "[apple-read-jpeg]") {
        U8Image halim = im::apple::read(
            "../tests/data/tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg");
    
        CHECK( halim.data() != nullptr );
        CHECK( halim.data() != 0 );
    }

    TEST_CASE("[apple] Check the dimensions of a new image", "[apple-image-dims]") {
        U8Image halim = im::apple::read(
            "../tests/data/tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg");
    
        WARN( "1.extent(0) = " << halim.extent(0) );
        WARN( "1.extent(1) = " << halim.extent(1) );
        WARN( "1.extent(2) = " << halim.extent(2) );
    
        WARN( "1.stride(0) = " << halim.stride(0) );
        WARN( "1.stride(1) = " << halim.stride(1) );
        WARN( "1.stride(2) = " << halim.stride(2) );
    
        U8Image halim2 = im::apple::read(
            "../tests/data/marci_512_srgb8.png");
    
        WARN( "2.extent(0) = " << halim2.extent(0) );
        WARN( "2.extent(1) = " << halim2.extent(1) );
        WARN( "2.extent(2) = " << halim2.extent(2) );
    
        WARN( "2.stride(0) = " << halim2.stride(0) );
        WARN( "2.stride(1) = " << halim2.stride(1) );
        WARN( "2.stride(2) = " << halim2.stride(2) );
    
    }
}