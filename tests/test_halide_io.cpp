
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/halide.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/IO/png.hh>
#include <libimread/IO/tiff.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

#define D(pth) "/Users/fish/Dropbox/libimread/tests/data/" pth
#define T(pth) "/tmp/" pth

namespace {
    
    using namespace Halide;
    using filesystem::path;
    using U8Image = im::HybridImage<uint8_t>;
    
    // static const bool collect_temporaries = true;
    #define COLLECT_TEMPORARIES 0
    #define CHECK_DIRECTORY "/Users/fish/Dropbox/libimread/check/"
    
    template <typename P>
    bool COLLECT(P&& p) {
        #if COLLECT_TEMPORARIES == 1
            path::makedir(CHECK_DIRECTORY);
            return path(std::forward<P>(p)).rename(CHECK_DIRECTORY);
        #else
            return path::remove(std::forward<P>(p));
        #endif
    }
    
    TEST_CASE("[halide-io] Read PNG files",
              "[halide-read-png]")
    {
        U8Image halim = im::halide::read(D("roses_512_rrt_srgb.png"));
        U8Image halim2 = im::halide::read(D("marci_512_srgb.png"));
        U8Image halim3 = im::halide::read(D("marci_512_srgb8.png"));
    }
    
    TEST_CASE("[halide-io] Read a PNG and rewrite it as a JPEG using tmpwrite()",
              "[halide-read-jpeg-write-png-tmpwrite]")
    {
        using im::format::JPG;
        
        U8Image halim = im::halide::read(D("roses_512_rrt_srgb.png"));
        auto tf = im::halide::tmpwrite<JPG>(halim);
        CHECK(COLLECT(tf));
        
        U8Image halim2 = im::halide::read(D("marci_512_srgb.png"));
        auto tf2 = im::halide::tmpwrite<JPG>(halim2);
        CHECK(COLLECT(tf2));
        
        U8Image halim3 = im::halide::read(D("marci_512_srgb8.png"));
        auto tf3 = im::halide::tmpwrite<JPG>(halim3);
        CHECK(COLLECT(tf3));
    }
    
    TEST_CASE("[halide-io] Read JPEG files",
              "[halide-read-jpeg]")
    {
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        U8Image halim2 = im::halide::read(D("IMG_4332.jpg"));
        U8Image halim3 = im::halide::read(D("IMG_7333.jpeg"));
    }
    
    TEST_CASE("[halide-io] Read a JPEG and rewrite it as a PNG using tmpwrite()",
              "[halide-read-jpeg-write-png-tmpwrite]")
    {
        using im::format::PNG;
        
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        auto tf = im::halide::tmpwrite<PNG>(halim);
        CHECK(COLLECT(tf));
        
        U8Image halim2 = im::halide::read(D("IMG_4332.jpg"));
        auto tf2 = im::halide::tmpwrite<PNG>(halim2);
        CHECK(COLLECT(tf2));
        
        U8Image halim3 = im::halide::read(D("IMG_7333.jpeg"));
        auto tf3 = im::halide::tmpwrite<PNG>(halim3);
        CHECK(COLLECT(tf3));
        
        U8Image halim4 = im::halide::read(D("10954288_342637995941364_1354507656_n.jpg"));
        auto tf4 = im::halide::tmpwrite<PNG>(halim4);
        CHECK(COLLECT(tf4));
    }
    
    TEST_CASE("[halide-io] Read TIFF files, rewrite each as a PNG using tmpwrite()",
              "[halide-read-tiff-write-png-tmpwrite]")
    {
        using im::format::PNG;
        
        path basedir(im::test::basedir);
        const std::vector<path> tifs = basedir.list("*.tif*");
        
        std::for_each(tifs.begin(), tifs.end(), [&basedir](path const& p) {
            auto tif = im::halide::read(basedir/p);
            auto pngpath = im::halide::tmpwrite<PNG>(tif);
            CHECK(COLLECT(pngpath));
        });
        
    }
    
    TEST_CASE("[halide-io] Read a JPEG and rewrite it as a TIFF using tmpwrite()",
              "[halide-read-jpeg-write-tiff-tmpwrite]")
    {
        using im::format::TIFF;
        
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        auto tf = im::halide::tmpwrite<TIFF>(halim);
        CHECK(COLLECT(tf));
        
        U8Image halim2 = im::halide::read(D("IMG_4332.jpg"));
        auto tf2 = im::halide::tmpwrite<TIFF>(halim2);
        CHECK(COLLECT(tf2));
        
        U8Image halim3 = im::halide::read(D("IMG_7333.jpeg"));
        auto tf3 = im::halide::tmpwrite<TIFF>(halim3);
        CHECK(COLLECT(tf3));
        
        U8Image halim4 = im::halide::read(D("10954288_342637995941364_1354507656_n.jpg"));
        auto tf4 = im::halide::tmpwrite<TIFF>(halim4);
        CHECK(COLLECT(tf4));
    }
    
    TEST_CASE("[halide-io] Read a TIFF, rewrite it as another TIFF using tmpwrite()",
              "[halide-read-tiff-write-tiff-tmpwrite]")
    {
        using im::format::TIFF;
        
        U8Image halim = im::halide::read(D("ptlobos.tif"));
        auto tf = im::halide::tmpwrite<TIFF>(halim);
        CHECK(COLLECT(tf));
    }
    
    TEST_CASE("[halide-io] Write multiple formats as PPM",
              "[halide-read-multiple-write-ppm]")
    {
        filesystem::TemporaryDirectory td("test-halide-io");
        
        U8Image halim = im::halide::read(D("ptlobos.tif"));
        im::halide::write(halim, td.dirpath/"PPM_DUG986.ppm");
        U8Image halim2 = im::halide::read(td.dirpath/"PPM_DUG986.ppm");
        im::halide::write(halim2, td.dirpath/"PPM_YO_DOGG222.png");
        
        U8Image halim3 = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        im::halide::write(halim3, td.dirpath/"PPM_OH_DOGGGGG.ppm");
        U8Image halim4 = im::halide::read(td.dirpath/"PPM_OH_DOGGGGG.ppm");
        im::halide::write(halim4, td.dirpath/"PPM_IMG_DOGGGGGGGGG.png");
        
        U8Image halim5 = im::halide::read(D("RGB888.pvr"));
        im::halide::write(halim5, td.dirpath/"POWERDOGGGGG.png");
    }
    
    TEST_CASE("[halide-io] Write multiple formats as TIFF",
              "[halide-read-multiple-write-tiff]")
    {
        filesystem::TemporaryDirectory td("test-halide-io");
        
        U8Image halim = im::halide::read(D("ptlobos.tif"));
        im::halide::write(halim, td.dirpath/"TIFF_DUG986.tiff");
        
        U8Image halim2 = im::halide::read(td.dirpath/"TIFF_DUG986.tiff");
        im::halide::write(halim2, td.dirpath/"TIFF_YO_DOGG222.tif");
        
        U8Image halim3 = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        im::halide::write(halim3, td.dirpath/"TIFF_OH_DOGGGGG.tiff");
        
        U8Image halim4 = im::halide::read(td.dirpath/"TIFF_OH_DOGGGGG.tiff");
        im::halide::write(halim4, td.dirpath/"TIFF_IMG_DOGGGGGGGGG.tif");
        
        U8Image halim5 = im::halide::read(D("RGB888.pvr"));
        im::halide::write(halim5, td.dirpath/"POWERDOGGGGG.tif");
    }
    
    /// LSM WRITING NOT IMPLEMENTED
    
    // TEST_CASE("[halide-io] Write multiple formats as LSM",
    //           "[halide-read-multiple-write-lsm]")
    // {
    //     filesystem::TemporaryDirectory td("test-halide-io");
    //
    //     U8Image halim = im::halide::read(D("ptlobos.tif"));
    //     im::halide::write(halim, td.dirpath/"TIFF_DUG986.lsm");
    //     U8Image halim2 = im::halide::read(td.dirpath/"TIFF_DUG986.tiff");
    //     im::halide::write(halim2, td.dirpath/"TIFF_YO_DOGG222.lsm");
    //
    //     U8Image halim3 = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
    //     im::halide::write(halim3, td.dirpath/"TIFF_OH_DOGGGGG.lsm");
    //     U8Image halim4 = im::halide::read(td.dirpath/"TIFF_OH_DOGGGGG.tiff");
    //     im::halide::write(halim4, td.dirpath/"TIFF_IMG_DOGGGGGGGGG.lsm");
    // }
    
    TEST_CASE("[halide-io] Check the dimensions of an image",
              "[halide-read-jpg-png-image-dims]")
    {
        U8Image halim = im::halide::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        U8Image halim2 = im::halide::read(D("marci_512_srgb8.png"));
    }
    
}