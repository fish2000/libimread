
#include <libimread/libimread.hpp>
#include <libimread/coregraphics.hh>
#include <libimread/fs.hh>

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
    
    TEST_CASE("[apple-io] Read a JPEG and rewrite it as a PNG via image_io.h", "[apple-read-jpeg-write-png]") {
        im::fs::TemporaryDirectory td("test-apple-io-XXXXX");
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        ext::save(halim, td.dirpath/"apple_YO_DOGG222.png");
    }
    
    TEST_CASE("[apple-io] Read a PNG", "[apple-read-png]") {
        im::fs::TemporaryDirectory td("test-apple-io-XXXXX");
        U8Image halim = im::apple::read(D("IMG_7333.jpeg"));
        ext::save(halim, td.dirpath/"apple_OH_DAWG666.png");
    }
    
    TEST_CASE("[apple-io] Read a TIFF", "[apple-read-tiff]") {
        im::fs::TemporaryDirectory td("test-apple-io-XXXXX");
        U8Image halim = im::apple::read(D("ptlobos.tif"));
        ext::save(halim, td.dirpath/"apple_TIFF_DUG986.png");
    }
    
    TEST_CASE("[apple-io] Read a JPEG", "[apple-read-jpeg]") {
        im::fs::TemporaryDirectory td("test-apple-io-XXXXX");
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        CHECK( halim.data() != nullptr );
        CHECK( halim.data() != 0 );
    }
    
    TEST_CASE("[apple-io] Check the dimensions of a new image", "[apple-image-dims]") {
        im::fs::TemporaryDirectory td("test-apple-io-XXXXX");
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        U8Image halim2 = im::apple::read(D("IMG_4332.jpg"));
    }
}