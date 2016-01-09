
#include <libimread/libimread.hpp>
#include <libimread/coregraphics.hh>
#include <libimread/interleaved.hh>
#include <libimread/fs.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

#define D(pth) "/Users/fish/Dropbox/libimread/tests/data/" pth
#define T(pth) "/tmp/" pth

namespace {
    
    using U8ImagePtr = im::apple::image_ptr<uint8_t>;
    using U8Image = im::apple::ImageType<uint8_t>;
    
    namespace ext {
        using namespace Halide;
        #include <libimread/private/image_io.h>
        
        template <typename T = uint8_t>
        void save_ptr(U8ImagePtr &&im, std::string fname) {
            ext::save(dynamic_cast<Image<T>>(*im), fname);
        }
    }
    
    TEST_CASE("[apple-io] Read a JPEG and rewrite it as a PNG via image_io.h",
              "[apple-read-jpeg-write-png]") {
        im::fs::TemporaryDirectory td("test-apple-io");
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        ext::save(halim, td.dirpath/"apple_YO_DOGG222.png");
    }
    
    TEST_CASE("[apple-io] Read a PNG and rewrite it as a PNG via image_io.h",
              "[apple-read-png]") {
        im::fs::TemporaryDirectory td("test-apple-io");
        U8Image halim = im::apple::read(D("IMG_7333.jpeg"));
        ext::save(halim, td.dirpath/"apple_OH_DAWG666.png");
    }
    
    TEST_CASE("[apple-io] Read a TIFF and rewrite it as a PNG via image_io.h",
              "[apple-read-tiff]") {
        im::fs::TemporaryDirectory td("test-apple-io");
        U8Image halim = im::apple::read(D("ptlobos.tif"));
        ext::save(halim, td.dirpath/"apple_TIFF_DUG986.png");
    }
    
    TEST_CASE("[apple-io] Read a JPEG",
              "[apple-read-jpeg]") {
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        // CHECK( halim.data() != nullptr );
        // CHECK( halim.data() != 0 );
    }
    
    TEST_CASE("[apple-io] Check dimensions of JPEG images",
              "[apple-image-jpeg-dims]") {
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        U8Image halim2 = im::apple::read(D("IMG_4332.jpg"));
    }
    
    using filesystem::path;
    
    TEST_CASE("[apple-io] Read PNG and JPEG files and write as PNGs with im::apple::write()",
              "[apple-read-png-jpeg-write-jpegs-with-im-apple-write]")
    {
        im::fs::TemporaryDirectory td("test-apple-io");
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        const std::vector<path> jpgs = basedir.list("*.jpg");
        
        std::for_each(pngs.begin(), pngs.end(), [&basedir, &td](const path &p) {
            auto png = im::apple::read(basedir/p);
            // WTF("Read image file:", (basedir/p).str(),
            //     FF("Image size: %i x %i x %i", png.dim(0), png.dim(1), png.dim(2)));
            path np = td.dirpath/p;
            path npext = np + ".png";
            im::apple::write(png, npext);
        });
        
        std::for_each(jpgs.begin(), jpgs.end(), [&basedir, &td](const path &p) {
            auto jpg = im::apple::read(basedir/p);
            path np = td.dirpath/p;
            path npext = np + ".png";
            im::apple::write(jpg, npext);
        });
        
        const std::vector<path> apples = td.dirpath.list("*.png");
        CHECK(apples.size() == pngs.size() + jpgs.size());
        
        std::for_each(apples.begin(), apples.end(), [&basedir, &td](const path &p) {
            path np = td.dirpath/p;
            auto mcintosh = im::halide::read(np);
            REQUIRE(mcintosh.dim(0) > 0);
            REQUIRE(mcintosh.dim(1) > 0);
        });
        
    }
    
}