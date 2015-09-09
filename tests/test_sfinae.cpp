
#include <libimread/libimread.hpp>
#include <libimread/base.hh>

#include <libimread/IO/png.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/IO/webp.hh>
#include <libimread/IO/pvrtc.hh>

using namespace im;

#include "include/catch.hpp"

TEST_CASE("[SFINAE] Check if PNG can read", "[sfinae-PNG-check-can-read]") {
    REQUIRE(has_read<format::PNG>());
}

TEST_CASE("[SFINAE] Check if JPEG can read", "[sfinae-JPEG-check-can-read]") {
    REQUIRE(has_read<format::JPG>());
    REQUIRE(has_read<format::JPEG>());
}

TEST_CASE("[SFINAE] Confirm WebP can NOT write", "[sfinae-WebP-confirm-no-write]") {
    REQUIRE(!has_write<format::WebP>());
}

TEST_CASE("[SFINAE] Confirm PVR can NOT write", "[sfinae-PVR-confirm-no-write]") {
    REQUIRE(!has_write<format::PVR>());
    REQUIRE(!has_write<format::PVRTC>());
}

