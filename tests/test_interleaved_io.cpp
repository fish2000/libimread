
#include <libimread/libimread.hpp>
#include <libimread/interleaved.hh>

#include "include/catch.hpp"

#define D(pth) "/Users/fish/Dropbox/libimread/tests/data/" pth
#define T(pth) "/tmp/" pth

namespace {
    
    using RGB = im::color::RGB;
    using RGBA = im::color::RGBA;
    using IRGB = im::InterleavedImage<RGB>;
    using IRGBA = im::InterleavedImage<RGBA>;
    
    im::fs::TemporaryDirectory td("test-interleaved-io-XXXXX");
    
    TEST_CASE("[interleaved-io] Read PNG files", "[interleaved-read-png]") {
        IRGBA leavedim = im::interleaved::read(D("roses_512_rrt_srgb.png"));
        IRGBA leavedim2 = im::interleaved::read(D("marci_512_srgb.png"));
        IRGBA leavedim3 = im::interleaved::read(D("marci_512_srgb8.png"));
    }
    
    TEST_CASE("[interleaved-io] Read a PNG and rewrite it as a JPEG", "[interleaved-read-jpeg-write-png]") {
        IRGBA leavedim = im::interleaved::read(D("roses_512_rrt_srgb.png"));
        im::interleaved::write(leavedim, td.dirpath/"jpgg_YO_DOGG222.jpg");
        
        IRGBA leavedim2 = im::interleaved::read(D("marci_512_srgb.png"));
        im::interleaved::write(leavedim2, td.dirpath/"jpgg_marci_512_srgb_YO.jpg");
        
        IRGBA leavedim3 = im::interleaved::read(D("marci_512_srgb8.png"));
        im::interleaved::write(leavedim3, td.dirpath/"jpgg_marci_512_srgb_YO_YO_YO.jpg");
    }
    
    TEST_CASE("[interleaved-io] Read JPEG files", "[interleaved-read-jpeg]") {
        IRGBA leavedim = im::interleaved::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        IRGBA leavedim2 = im::interleaved::read(D("IMG_4332.jpg"));
        IRGBA leavedim3 = im::interleaved::read(D("IMG_7333.jpeg"));
    }
    
    // TEST_CASE("[interleaved-io] Read a JPEG and rewrite it as a PNG", "[interleaved-read-jpeg-write-png]") {
    //     IRGBA leavedim = im::interleaved::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
    //     im::interleaved::write(leavedim, td.dirpath/"OH_DAWG666.png");
    //
    //     IRGBA leavedim2 = im::interleaved::read(D("IMG_4332.jpg"));
    //     im::interleaved::write(leavedim2, td.dirpath/"IMG_4332_JPG.png");
    //
    //     IRGBA leavedim3 = im::interleaved::read(D("IMG_7333.jpeg"));
    //     im::interleaved::write(leavedim3, td.dirpath/"IMG_7333_JPG.png");
    //
    //     IRGBA leavedim4 = im::interleaved::read(D("10954288_342637995941364_1354507656_n.jpg"));
    //     im::interleaved::write(leavedim4, td.dirpath/"HAY_GUISE.png");
    // }
    
    // TEST_CASE("[interleaved-io] Read a TIFF", "[interleaved-read-tiff]") {
    //     IRGBA leavedim = im::interleaved::read(D("ptlobos.tif"));
    //     im::interleaved::write(leavedim, td.dirpath/"TIFF_DUG986.png");
    // }
    
    // TEST_CASE("[interleaved-io] Write multiple formats as PPM", "[interleaved-read-tiff-write-ppm]") {
    //     IRGBA leavedim = im::interleaved::read(D("ptlobos.tif"));
    //     im::interleaved::write(leavedim, td.dirpath/"PPM_DUG986.ppm");
    //     IRGBA leavedim2 = im::interleaved::read(td.dirpath/"PPM_DUG986.ppm");
    //     im::interleaved::write(leavedim2, td.dirpath/"PPM_YO_DOGG222.png");
    //
    //     IRGBA leavedim3 = im::interleaved::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
    //     im::interleaved::write(leavedim3, td.dirpath/"PPM_OH_DOGGGGG.ppm");
    //     IRGBA leavedim4 = im::interleaved::read(td.dirpath/"PPM_OH_DOGGGGG.ppm");
    //     im::interleaved::write(leavedim4, td.dirpath/"PPM_IMG_DOGGGGGGGGG.png");
    // }
    
    TEST_CASE("[interleaved-io] Check the dimensions of a new image", "[interleaved-image-dims]") {
        IRGBA leavedim = im::interleaved::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        IRGBA leavedim2 = im::interleaved::read(D("marci_512_srgb8.png"));
    }
}