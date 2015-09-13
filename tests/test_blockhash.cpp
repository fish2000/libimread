
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/hashing.hh>
#include <libimread/fs.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    
    TEST_CASE("[blockhash] Calculate blockhash_quick for HybridImage PNG instances",
              "[blockhash-quick-hybridimage-png]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        std::for_each(pngs.begin(), pngs.end(), [&basedir](const path &p) {
            auto png = im::halide::read(basedir/p);
            auto bithash = blockhash::blockhash_quick(png);
            WTF("BLOCKHASH_QUICK:", bithash.to_ullong(),
                                    bithash.to_string());
        });
    }
    
    TEST_CASE("[blockhash] Calculate blockhash for HybridImage PNG instances",
              "[blockhash-hybridimage-png]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        std::for_each(pngs.begin(), pngs.end(), [&basedir](const path &p) {
            auto png = im::halide::read(basedir/p);
            auto bithash = blockhash::blockhash(png);
            WTF("BLOCKHASH:", bithash.to_ullong(),
                              bithash.to_string());
        });
    }
    
};