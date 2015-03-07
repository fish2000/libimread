
#include <libimread/libimread.hpp>
#include <libimread/base.hh>

#include <libimread/_png.hh>

#include "include/catch.hpp"

TEST_CASE("Check if PNG can read") {
    REQUIRE(im::has_read<im::format::PNG>());
}
