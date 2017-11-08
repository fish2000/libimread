
#define CATCH_CONFIG_FAST_COMPILE

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/hashing.hh>
#include <libimread/ext/filesystem/path.h>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using pathvec_t = std::vector<path>;
    
    namespace detail {
        /// to_hex() courtesy of:
        /// http://stackoverflow.com/a/5100745/298171
        template <typename T> inline
        std::string to_hex(T tvalue) {
            std::stringstream stream;
            stream << "0x"
                   << std::setfill('0') << std::setw(sizeof(T) * 2)
                   << std::hex << tvalue;
            return stream.str();
        }
    }
    
    TEST_CASE("[blockhash] Calculate blockhash_quick for HybridImage PNG instances",
              "[blockhash-quick-hybridimage-png]")
    {
        path basedir(im::test::basedir);
        const pathvec_t pngs = basedir.list("*.png");
        std::for_each(pngs.begin(), pngs.end(), [&basedir](path const& p) {
            auto png = im::halide::read(basedir/p);
            auto bithash = blockhash::blockhash_quick(png);
            unsigned long long longhash = bithash.to_ullong();
            std::string hexhash = blockhash::detail::hexify(bithash);
            // CHECK(hexhash == detail::to_hex(longhash));
            // WTF("BLOCKHASH_QUICK:", hexhash,
            //                         bithash.to_string());
        });
    }
    
    TEST_CASE("[blockhash] Calculate blockhash for HybridImage PNG instances",
              "[blockhash-hybridimage-png]")
    {
        path basedir(im::test::basedir);
        const pathvec_t pngs = basedir.list("*.png");
        std::for_each(pngs.begin(), pngs.end(), [&basedir](path const& p) {
            auto png = im::halide::read(basedir/p);
            auto bithash = blockhash::blockhash(png);
            unsigned long long longhash = bithash.to_ullong();
            std::string hexhash = blockhash::detail::hexify(bithash);
            // CHECK(hexhash == detail::to_hex(longhash));
            // WTF("BLOCKHASH:", hexhash,
            //                   bithash.to_string());
        });
    }
    
}