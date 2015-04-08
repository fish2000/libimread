
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/halide.hh>

#include "include/catch.hpp"

#define D(pth) "../tests/data/" pth
#define T(pth) "/tmp/" pth

namespace {
    
    using namespace Halide;
    using U8Image = im::HybridImage<uint8_t>;
    
    namespace ext {
        #include <libimread/private/image_io.h>
    }
    
    TEST_CASE("Read PNG files", "[read-png]") {
        U8Image halim = im::halide::read(D("roses_512_rrt_srgb.png"));
        U8Image halim2 = im::halide::read(D("marci_512_srgb.png"));
        U8Image halim3 = im::halide::read(D("marci_512_srgb8.png"));
    }
    
    TEST_CASE("Read a PNG and rewrite it via image_io.h", "[read-jpeg-write-png]") {
        U8Image halim = im::halide::read(D("roses_512_rrt_srgb.png"));
        im::halide::write(halim, T("YO_DOGG222.png"));
        
        U8Image halim2 = im::halide::read(D("marci_512_srgb.png"));
        im::halide::write(halim2, T("marci_512_srgb_YO.png"));
        
        U8Image halim3 = im::halide::read(D("marci_512_srgb8.png"));
        im::halide::write(halim3, T("marci_512_srgb_YO_YO_YO.png"));
    }
    
    TEST_CASE("Read JPEG files", "[read-jpeg]") {
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        U8Image halim2 = im::halide::read(D("IMG_4332.jpg"));
        U8Image halim3 = im::halide::read(D("IMG_7333.jpeg"));
    }
    
    TEST_CASE("Read a JPEG and rewrite it as a PNG via image_io.h", "[read-jpeg-write-png]") {
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        im::halide::write(halim, T("OH_DAWG666.png"));
        
        U8Image halim2 = im::halide::read(D("IMG_4332.jpg"));
        im::halide::write(halim2, T("IMG_4332_JPG.png"));
        
        U8Image halim3 = im::halide::read(D("IMG_7333.jpeg"));
        im::halide::write(halim3, T("IMG_7333_JPG.png"));
        
        U8Image halim4 = im::halide::read(D("10954288_342637995941364_1354507656_n.jpg"));
        im::halide::write(halim4, T("HAY_GUISE.png"));
    }
    
    TEST_CASE("Read a TIFF", "[read-tiff]") {
        U8Image halim = im::halide::read(D("ptlobos.tif"));
        im::halide::write(halim, T("TIFF_DUG986.png"));
    }
    
    TEST_CASE("Write multiple formats as PPM", "[read-tiff-write-ppm]") {
        U8Image halim = im::halide::read(D("ptlobos.tif"));
        im::halide::write(halim, T("PPM_DUG986.ppm"));
        U8Image halim2 = im::halide::read(T("PPM_DUG986.ppm"));
        im::halide::write(halim2, T("PPM_YO_DOGG222.png"));
        
        REQUIRE(halim.ndims() == halim2.ndims());
        REQUIRE(halim.stride(0) == halim2.stride(0));
        REQUIRE(halim.stride(1) == halim2.stride(1));
        REQUIRE(halim.stride(2) == halim2.stride(2));
        REQUIRE(halim.extent(0) == halim2.extent(0));
        REQUIRE(halim.extent(1) == halim2.extent(1));
        REQUIRE(halim.extent(2) == halim2.extent(2));
        
        U8Image halim3 = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        im::halide::write(halim3, T("PPM_OH_DOGGGGG.ppm"));
        U8Image halim4 = im::halide::read(T("PPM_OH_DOGGGGG.ppm"));
        im::halide::write(halim4, T("PPM_IMG_DOGGGGGGGGG.png"));
        
        REQUIRE(halim3.ndims() == halim4.ndims());
        REQUIRE(halim3.stride(0) == halim4.stride(0));
        REQUIRE(halim3.stride(1) == halim4.stride(1));
        REQUIRE(halim3.stride(2) == halim4.stride(2));
        REQUIRE(halim3.extent(0) == halim4.extent(0));
        REQUIRE(halim3.extent(1) == halim4.extent(1));
        REQUIRE(halim3.extent(2) == halim4.extent(2));
        
    }
    
    TEST_CASE("Check the dimensions of a new image", "[image-dims]") {
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
    
        WARN( "1.extent(0) = " << halim.extent(0) );
        WARN( "1.extent(1) = " << halim.extent(1) );
        WARN( "1.extent(2) = " << halim.extent(2) );
    
        WARN( "1.stride(0) = " << halim.stride(0) );
        WARN( "1.stride(1) = " << halim.stride(1) );
        WARN( "1.stride(2) = " << halim.stride(2) );
    
        U8Image halim2 = im::halide::read(D("marci_512_srgb8.png"));
    
        WARN( "2.extent(0) = " << halim2.extent(0) );
        WARN( "2.extent(1) = " << halim2.extent(1) );
        WARN( "2.extent(2) = " << halim2.extent(2) );
    
        WARN( "2.stride(0) = " << halim2.stride(0) );
        WARN( "2.stride(1) = " << halim2.stride(1) );
        WARN( "2.stride(2) = " << halim2.stride(2) );
    
    }
}