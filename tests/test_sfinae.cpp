
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/_png.hh>
using namespace im;

#include "include/catch.hpp"

TEST_CASE("Check if PNG can read") {
    REQUIRE(has_read<format::PNG>());
}
