
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/hashing.hh>
#include <libimread/fs.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    
    TEST_CASE("[blockhash] Calculate blockhash for HybridImage instances",
              "[blockhash-calculate-for-hybridimage-instances]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        std::for_each(pngs.begin(), pngs.end(), [&basedir](const path &p) {
            auto png = im::halide::read(basedir/p);
            REQUIRE(png.width() > 0);
            REQUIRE(png.height() > 0);
            auto bithash = blockhash::blockhash_quick(png);
            WTF("BLOCKHASH:", bithash.to_string());
        });
        
    }
    
};