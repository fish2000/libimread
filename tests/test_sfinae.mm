
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/objc-rt/objc-rt.hh>

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
    using objc_object_t = struct objc_object;
    struct non_object {};
    
    bool check_one = objc::traits::detail::has_isa<objc_object_t>::value == true;
    bool check_one_and_a_half = objc::traits::detail::is_object_pointer<NSObject*>::value == true;
    bool check_two = objc::traits::is_object<objc::types::ID>::value == true;
    bool check_three = objc::traits::is_object<non_object*>::value == false;
    CHECK(check_one);
    CHECK(check_one_and_a_half);
    CHECK(check_two);
    CHECK(check_three);
}

TEST_CASE("[SFINAE] Confirm results of objc::traits::is_object<NSObject*>::value",
          "[sfinae-objc-traits-confirm-NSObject-pointer-value]") {
    bool check_one = objc::traits::detail::has_isa<objc_object>::value == true;
    bool check_two = objc::traits::is_object<NSObject*>::value == true;
    CHECK(check_one);
    CHECK(check_two);
}

