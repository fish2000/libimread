
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/halide.hh>

#include "include/catch.hpp"

namespace {
    
    using namespace Halide;
    using U8Image = Image<uint8_t>;
    
    namespace ext {
        #include <libimread/private/image_io.h>
    }
    
    TEST_CASE("Read PNG files", "[read-png]") {
        U8Image halim = im::halide::read(
            "../tests/data/roses_512_rrt_srgb.png");
        U8Image halim2 = im::halide::read(
            "../tests/data/marci_512_srgb.png");
        U8Image halim3 = im::halide::read(
            "../tests/data/marci_512_srgb8.png");
    }
    
    TEST_CASE("Read a PNG and rewrite it via image_io.h", "[read-jpeg-write-png]") {
        U8Image halim = im::halide::read(
            "../tests/data/roses_512_rrt_srgb.png");
        ext::save(halim, "/tmp/YO_DOGG222.png");
        U8Image halim2 = im::halide::read(
            "../tests/data/marci_512_srgb.png");
        ext::save(halim2, "/tmp/marci_512_srgb_YO.png");
        U8Image halim3 = im::halide::read(
            "../tests/data/marci_512_srgb8.png");
        ext::save(halim3, "/tmp/marci_512_srgb_YO_YO_YO.png");
    }
    
    TEST_CASE("Read JPEG files", "[read-jpeg]") {
        U8Image halim = im::halide::read(
            "../tests/data/tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg");
        U8Image halim2 = im::halide::read(
            "../tests/data/IMG_4332.jpg");
        U8Image halim3 = im::halide::read(
            "../tests/data/IMG_7333.jpeg");
    }
    
    TEST_CASE("Read a JPEG and rewrite it as a PNG via image_io.h", "[read-jpeg-write-png]") {
        U8Image halim = im::halide::read(
            "../tests/data/tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg");
        ext::save(halim, "/tmp/OH_DAWG666.png");
        U8Image halim2 = im::halide::read(
            "../tests/data/IMG_4332.jpg");
        ext::save(halim2, "/tmp/IMG_4332_JPG.png");
        U8Image halim3 = im::halide::read(
            "../tests/data/IMG_7333.jpeg");
        ext::save(halim3, "/tmp/IMG_7333_JPG.png");
        U8Image halim4 = im::halide::read(
            "../tests/data/10954288_342637995941364_1354507656_n.jpg");
        ext::save(halim4, "/tmp/HAY_GUISE.png");
    }
    
    /*
    
    TEST_CASE("Read a TIFF", "[read-tiff]") {
        U8Image halim = im::halide::read(
            "../tests/data/ptlobos.tif");
        ext::save(halim, "/tmp/TIFF_DUG986.png");
    }
    
    */
    
    TEST_CASE("Check the dimensions of a new image", "[image-dims]") {
        U8Image halim = im::halide::read(
            "../tests/data/tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg");
    
        WARN( "1.extent(0) = " << halim.extent(0) );
        WARN( "1.extent(1) = " << halim.extent(1) );
        WARN( "1.extent(2) = " << halim.extent(2) );
    
        WARN( "1.stride(0) = " << halim.stride(0) );
        WARN( "1.stride(1) = " << halim.stride(1) );
        WARN( "1.stride(2) = " << halim.stride(2) );
    
        U8Image halim2 = im::halide::read(
            "../tests/data/marci_512_srgb8.png");
    
        WARN( "2.extent(0) = " << halim2.extent(0) );
        WARN( "2.extent(1) = " << halim2.extent(1) );
        WARN( "2.extent(2) = " << halim2.extent(2) );
    
        WARN( "2.stride(0) = " << halim2.stride(0) );
        WARN( "2.stride(1) = " << halim2.stride(1) );
        WARN( "2.stride(2) = " << halim2.stride(2) );
    
    }
}