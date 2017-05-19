
#include <string>
#include <vector>
#include <algorithm>
#include <regex>

#include <libimread/libimread.hpp>
#include <libimread/interleaved.hh>
#include <libimread/ext/filesystem/path.h>

#include "include/test_data.hpp"
#include "include/catch.hpp"

#define D(pth) "/Users/fish/Dropbox/libimread/tests/data/" pth
#define T(pth) "/tmp/" pth

namespace {
    
    using filesystem::path;
    using pathvec_t = std::vector<path>;
    
    // template <typename Color = color::RGBA>
    // InterleavedImage<Color>
    
    using RGB = im::color::RGB;
    using RGBA = im::color::RGBA;
    using IRGB = im::InterleavedImage<RGB>;
    using IRGBA = im::InterleavedImage<RGBA>;
    
    // using IRGB = std::unique_ptr<im::Image>;
    // using IRGBA = std::unique_ptr<im::Image>;
    
    using MetaRGB = im::Meta<RGB, 3>;
    using MetaRGBA = im::Meta<RGBA, 3>;
    using Index = im::Index<3>;
    
    TEST_CASE("[interleaved-io] Test Meta and Index object attributes",
              "[interleaved-meta-index-object-attributes]")
    {
        MetaRGB meta = MetaRGB(1024, 768);
        Index idx0 { 0, 1, 2 };
        REQUIRE(meta.contains(idx0));
        Index idx1 { 0, 1 };
        REQUIRE(meta.contains(idx1));
        Index idx2 = idx0 + idx1;
        REQUIRE(meta.contains(idx2));
    }
    
    TEST_CASE("[interleaved-io] Read PNG files",
              "[interleaved-read-png]")
    {
        path basedir(im::test::basedir);
        const pathvec_t pngs = basedir.list("*.png");
        std::for_each(pngs.begin(), pngs.end(), [&basedir](path const& p) {
            auto png = im::interleaved::read(basedir/p);
            REQUIRE(png.width() > 0);
            REQUIRE(png.height() > 0);
        });
    }
    
    TEST_CASE("[interleaved-io] Read JPEG files",
              "[interleaved-read-jpg]")
    {
        path basedir(im::test::basedir);
        const pathvec_t jpgs = basedir.list("*.jpg");
        std::for_each(jpgs.begin(), jpgs.end(), [&basedir](path const& p) {
            auto jpg = im::interleaved::read(basedir/p);
            CHECK(jpg.width() > 0);
            CHECK(jpg.height() > 0);
        });
    }
    
}