
#include <string>
#include <libimread/libimread.hpp>
#include <libimread/ext/glob.hh>
#include "include/catch.hpp"

namespace {
    
    TEST_CASE("[glob-stringview] Test glob::match() and glob::imatch per the original testsuite",
              "[glob-stringview-test-match-imatch-per-original-testsuite]")
    {
        CHECK(glob::match("hello", "hello"));
        CHECK(!glob::match("hello", "hello!"));
        CHECK(!glob::match("hello", "hi"));
        
        CHECK(glob::match("he?lo", "hello"));
        // CHECK(glob::match("h*o", "hello"));
        // CHECK(glob::match("h******o", "hello"));
        // CHECK(glob::match("h***?***o", "hello"));
        // CHECK(glob::match("*o", "hello"));
        CHECK(glob::match("h*", "hello"));
        
        CHECK(!glob::match("", "hello"));
        CHECK(glob::match("", ""));
        CHECK(glob::match("*", ""));
        CHECK(glob::match("*", "hello"));
        CHECK(!glob::match("?", ""));
        
        // CHECK(glob::match(std::string("h***?***o"), std::string("hello")));
        
        CHECK(!glob::match("hello", "HELLO"));
        CHECK(glob::imatch("hello", "HELLO"));
        // CHECK(glob::imatch("h*L?", "hello"));
        CHECK(!glob::match("h*L?", "hello"));
    }
}
