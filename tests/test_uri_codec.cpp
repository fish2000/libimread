
#include <string>
#include <cstdlib>
#include <ctime>

// #include <libimread/libimread.hpp>
// #include <libimread/errors.hh>
#include <libimread/ext/uri.hh>

#include "include/catch.hpp"

namespace {
    
    using namespace im;
    
    /// Bundled tests adapted from inline Verizon test code found here:
    /// https://github.com/Verizon/hlx/blob/74ec6eb44eba4cf6b1a87e1b94a5741f2a08f3e6/src/core/support/uri.cc#L185-L235
    
    TEST_CASE("[uri] Run bundled encode/decode tests",
              "[uri-run-bundled-encode-decode-tests]")
    {
        /// basic encode/decode checks:
        CHECK(uri::encode("ABC") == "ABC");
        
        const std::string orig("\0\1\2", 3);
        const std::string enc("%00%01%02");
        CHECK(uri::encode(orig) == enc);
        CHECK(uri::decode(enc) == orig);
        
        CHECK(uri::encode("\xFF") == "%FF");
        CHECK(uri::decode("%FF") == "\xFF");
        CHECK(uri::decode("%ff") == "\xFF");
        
        /// unsafe chars test, RFC1738:
        const std::string unsafe(" <>#{}|\\^~[]`");
        std::string unsafe_encoded = uri::encode(unsafe);
        CHECK(std::string::npos == unsafe_encoded.find_first_of(unsafe));
        CHECK(uri::decode(unsafe_encoded) == unsafe);
        
        /// random-character test:
        const int MAX_LEN = 128;
        char a[MAX_LEN];
        int i = 0,
            j = 0;
        
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        
        for (; i < 100; ++i) {
            for (; j < MAX_LEN; ++j) {
                a[j] = std::rand() % (1 << 8);
            }
            int length = std::rand() % MAX_LEN;
            std::string original(a, length);
            std::string encoded = uri::encode(original);
            CHECK(original == uri::decode(encoded));
        }
        
    }
    
} /// namespace (anon.)
