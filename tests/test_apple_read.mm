
#include <libimread/libimread.hpp>
#include <libimread/coregraphics.hh>

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
    
    TEST_CASE("[apple] Read a JPEG and rewrite it as a PNG via image_io.h", "[apple-read-jpeg-write-png]") {
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        // ext::save_ptr(std::move(halim), "/tmp/apple_YO_DOGG222.png");
        ext::save(halim, T("apple_YO_DOGG222.png"));
    }
    
    TEST_CASE("[apple] Read a PNG", "[apple-read-png]") {
        U8Image halim = im::apple::read(D("IMG_7333.jpeg"));
        // ext::save_ptr(std::move(halim), "/tmp/apple_OH_DAWG666.png");
        ext::save(halim, T("apple_OH_DAWG666.png"));
    }
    
    TEST_CASE("[apple] Read a TIFF", "[apple-read-tiff]") {
        U8Image halim = im::apple::read(D("ptlobos.tif"));
        // ext::save_ptr(std::move(halim), "/tmp/apple_TIFF_DUG986.png");
        ext::save(halim, T("apple_TIFF_DUG986.png"));
    }
    
    TEST_CASE("[apple] Read a JPEG", "[apple-read-jpeg]") {
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        CHECK( halim.data() != nullptr );
        CHECK( halim.data() != 0 );
    }
    
    TEST_CASE("[apple] Check the dimensions of a new image", "[apple-image-dims]") {
        U8Image halim = im::apple::read(D("tumblr_mgq73sTl6z1qb9r7fo1_r1_500.jpg"));
        
        // WARN( "1.extent(0) = " << halim.extent(0) );
        // WARN( "1.extent(1) = " << halim.extent(1) );
        // WARN( "1.extent(2) = " << halim.extent(2) );
        //
        // WARN( "1.stride(0) = " << halim.stride(0) );
        // WARN( "1.stride(1) = " << halim.stride(1) );
        // WARN( "1.stride(2) = " << halim.stride(2) );
        
        U8Image halim2 = im::apple::read(D("IMG_4332.jpg"));
        
        // WARN( "2.extent(0) = " << halim2.extent(0) );
        // WARN( "2.extent(1) = " << halim2.extent(1) );
        // WARN( "2.extent(2) = " << halim2.extent(2) );
        //
        // WARN( "2.stride(0) = " << halim2.stride(0) );
        // WARN( "2.stride(1) = " << halim2.stride(1) );
        // WARN( "2.stride(2) = " << halim2.stride(2) );
    
    }
}