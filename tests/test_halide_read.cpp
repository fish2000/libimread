
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/halide.hh>

#include "include/catch.hpp"

#define D(pth) "/Users/fish/Dropbox/libimread/tests/data/" pth
#define T(pth) "/tmp/" pth

namespace {
    
    using namespace Halide;
    using U8Image = im::HybridImage<uint8_t>;
    
    namespace ext {
        #include <libimread/private/image_io.h>
    }
    
    im::fs::TemporaryDirectory td("test-halide-read-XXXXX");
    
    TEST_CASE("Read PNG files", "[read-png]") {
        U8Image halim = im::halide::read(D("roses_512_rrt_srgb.png"));
        U8Image halim2 = im::halide::read(D("marci_512_srgb.png"));
        U8Image halim3 = im::halide::read(D("marci_512_srgb8.png"));
    }
    
    TEST_CASE("Read a PNG and rewrite it as a JPEG", "[read-jpeg-write-png]") {
        U8Image halim = im::halide::read(D("roses_512_rrt_srgb.png"));
        im::halide::write(halim, td.dirpath/"jpgg_YO_DOGG222.jpg");
        
        U8Image halim2 = im::halide::read(D("marci_512_srgb.png"));
        im::halide::write(halim2, td.dirpath/"jpgg_marci_512_srgb_YO.jpg");
        
        U8Image halim3 = im::halide::read(D("marci_512_srgb8.png"));
        im::halide::write(halim3, td.dirpath/"jpgg_marci_512_srgb_YO_YO_YO.jpg");
    }
    
    TEST_CASE("Read JPEG files", "[read-jpeg]") {
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        U8Image halim2 = im::halide::read(D("IMG_4332.jpg"));
        U8Image halim3 = im::halide::read(D("IMG_7333.jpeg"));
    }
    
    TEST_CASE("Read a JPEG and rewrite it as a PNG", "[read-jpeg-write-png]") {
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        im::halide::write(halim, td.dirpath/"OH_DAWG666.png");
        
        U8Image halim2 = im::halide::read(D("IMG_4332.jpg"));
        im::halide::write(halim2, td.dirpath/"IMG_4332_JPG.png");
        
        U8Image halim3 = im::halide::read(D("IMG_7333.jpeg"));
        im::halide::write(halim3, td.dirpath/"IMG_7333_JPG.png");
        
        U8Image halim4 = im::halide::read(D("10954288_342637995941364_1354507656_n.jpg"));
        im::halide::write(halim4, td.dirpath/"HAY_GUISE.png");
    }
    
    TEST_CASE("Read a TIFF", "[read-tiff]") {
        U8Image halim = im::halide::read(D("ptlobos.tif"));
        im::halide::write(halim, td.dirpath/"TIFF_DUG986.png");
    }
    
    TEST_CASE("Write multiple formats as PPM", "[read-tiff-write-ppm]") {
        U8Image halim = im::halide::read(D("ptlobos.tif"));
        im::halide::write(halim, td.dirpath/"PPM_DUG986.ppm");
        U8Image halim2 = im::halide::read(td.dirpath/"PPM_DUG986.ppm");
        im::halide::write(halim2, td.dirpath/"PPM_YO_DOGG222.png");
        
        U8Image halim3 = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        im::halide::write(halim3, td.dirpath/"PPM_OH_DOGGGGG.ppm");
        U8Image halim4 = im::halide::read(td.dirpath/"PPM_OH_DOGGGGG.ppm");
        im::halide::write(halim4, td.dirpath/"PPM_IMG_DOGGGGGGGGG.png");
        
    }
    
    TEST_CASE("Check the dimensions of a new image", "[image-dims]") {
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        U8Image halim2 = im::halide::read(D("marci_512_srgb8.png"));
    }
}