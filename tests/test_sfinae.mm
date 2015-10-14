
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/objc-rt.hh>

#include <libimread/IO/png.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/IO/webp.hh>
#include <libimread/IO/pvrtc.hh>

#include "include/catch.hpp"

TEST_CASE("[SFINAE] Check if PNG can read",
          "[sfinae-PNG-check-can-read]")
{
    CHECK(im::has_read<im::format::PNG>());
}

TEST_CASE("[SFINAE] Check if JPEG can read",
          "[sfinae-JPEG-check-can-read]")
{
    CHECK(im::has_read<im::format::JPG>());
    CHECK(im::has_read<im::format::JPEG>());
}

TEST_CASE("[SFINAE] Confirm WebP can NOT write",
          "[sfinae-WebP-confirm-no-write]")
{
    CHECK(!im::has_write<im::format::WebP>());
}

TEST_CASE("[SFINAE] Confirm PVR can NOT write",
          "[sfinae-PVR-confirm-no-write]")
{
    CHECK(!im::has_write<im::format::PVR>());
    CHECK(!im::has_write<im::format::PVRTC>());
}

TEST_CASE("[SFINAE] Confirm results of objc::traits::is_object<objc::types::ID>::value",
          "[sfinae-objc-traits-confirm-objc-types-ID-value]") {
    CHECK(objc::traits::is_object<objc::types::ID>::value);
}

TEST_CASE("[SFINAE] Confirm results of objc::traits::is_object<NSObject*>::value",
          "[sfinae-objc-traits-confirm-NSObject-pointer-value]") {
    CHECK(objc::traits::is_object<NSObject*>::value);
}

