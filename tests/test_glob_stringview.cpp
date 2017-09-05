
#include <vector>
#include <string>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/ext/glob.hh>

#include "include/catch.hpp"

namespace {
    
    TEST_CASE("[glob-stringview] Test glob::match() and glob::imatch() per the original testsuite",
              "[glob-stringview-test-match-imatch-per-original-testsuite]")
    {
        CHECK(glob::match("hello", "hello"));
        CHECK(!glob::match("hello", "hello!"));
        CHECK(!glob::match("hello", "hi"));
        
        CHECK(glob::match("he?lo", "hello"));
        CHECK(glob::match("h*o", "hello"));
        CHECK(glob::match("h******o", "hello"));
        CHECK(glob::match("h***?***o", "hello"));
        CHECK(glob::match("*o", "hello"));
        CHECK(glob::match("h*", "hello"));
        
        CHECK(!glob::match("", "hello"));
        CHECK(glob::match("", ""));
        CHECK(glob::match("*", ""));
        CHECK(glob::match("*", "hello"));
        CHECK(!glob::match("?", ""));
        
        CHECK(glob::match(std::string("h***?***o"), std::string("hello")));
        
        CHECK(!glob::match("hello", "HELLO"));
        CHECK(glob::imatch("hello", "HELLO"));
        CHECK(glob::imatch("h*L?", "hello"));
        CHECK(!glob::match("h*L?", "hello"));
    }
    
    std::vector<std::string> strings = {
        "yo", "dogg", "hello", "there", "i", "heard", "you", "like", "globs"
    };
    
    TEST_CASE("[glob-stringview] Test glob::match() and glob::imatch() on string vectors using iterators",
              "[glob-stringview-test-match-imatch-on-string-vectors-using-iterators]")
    {
        CHECK(std::count_if(strings.begin(), strings.end(),
                         [](std::string const& s) { return glob::match("*o", s); }) == 2);
        CHECK(std::count_if(strings.begin(), strings.end(),
                         [](std::string const& s) { return glob::match("*o*", s); }) == 5);
        CHECK(std::count_if(strings.begin(), strings.end(),
                         [](std::string const& s) { return glob::match("yo*", s); }) == 2);
        CHECK(std::count_if(strings.begin(), strings.end(),
                         [](std::string const& s) { return glob::match("*i*", s); }) == 2);
        
        CHECK(std::count_if(strings.begin(), strings.end(), glob::matcher("*o")) == 2);
        CHECK(std::count_if(strings.begin(), strings.end(), glob::matcher("*o*")) == 5);
        CHECK(std::count_if(strings.begin(), strings.end(), glob::matcher("yo*")) == 2);
        CHECK(std::count_if(strings.begin(), strings.end(), glob::matcher("*i*")) == 2);
    }
    
}
