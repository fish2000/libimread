
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/errors/demangle.hh>
#include <libimread/image.hh>
#include <libimread/interleaved.hh>
#include <libimread/base.hh>

#include <libimread/IO/png.hh>
#include <libimread/IO/jpeg.hh>
#include <libimread/IO/webp.hh>
#include <libimread/IO/pvrtc.hh>

#include "include/catch.hpp"

TEST_CASE("[SFINAE] Check if PNG can read",
          "[sfinae-PNG-check-can-read]")
{
    CHECK(im::traits::has_read<im::format::PNG>());
    CHECK(im::traits::has_write<im::format::PNG>());
    CHECK(im::format::PNG::capacity.can_read);
    CHECK(im::format::PNG::capacity.can_write);
}

TEST_CASE("[SFINAE] Check if JPEG can read",
          "[sfinae-JPEG-check-can-read]")
{
    CHECK(im::traits::has_read<im::format::JPG>());
    CHECK(im::traits::has_read<im::format::JPEG>());
    CHECK(im::traits::has_write<im::format::JPG>());
    CHECK(im::traits::has_write<im::format::JPEG>());
    CHECK(im::format::JPG::capacity.can_read);
    CHECK(im::format::JPEG::capacity.can_read);
    CHECK(im::format::JPG::capacity.can_write);
    CHECK(im::format::JPEG::capacity.can_write);
}

TEST_CASE("[SFINAE] Confirm WebP can NOT write",
          "[sfinae-WebP-confirm-no-write]")
{
    CHECK(im::traits::has_read<im::format::WebP>());
    CHECK(im::format::WebP::capacity.can_read);
    CHECK(!im::traits::has_write<im::format::WebP>());
    CHECK(!im::format::WebP::capacity.can_write);
}

TEST_CASE("[SFINAE] Confirm PVR can NOT write",
          "[sfinae-PVR-confirm-no-write]")
{
    CHECK(im::traits::has_read<im::format::PVR>());
    CHECK(im::traits::has_read<im::format::PVRTC>());
    CHECK(!im::traits::has_write<im::format::PVR>());
    CHECK(!im::traits::has_write<im::format::PVRTC>());
    CHECK(im::format::PVR::capacity.can_read);
    CHECK(im::format::PVRTC::capacity.can_read);
    CHECK(!im::format::PVR::capacity.can_write);
    CHECK(!im::format::PVRTC::capacity.can_write);
}

// TEST_CASE("[SFINAE] Confirm results of objc::traits::is_object<objc::types::ID>::value",
//           "[sfinae-objc-traits-confirm-objc-types-ID-value]") {
//     using objc_object_t = struct objc_object;
//     struct non_object {};
//
//     bool check_one = objc::traits::detail::has_isa<objc_object_t>::value == true;
//     bool check_one_and_a_half = objc::traits::detail::is_object_pointer<NSObject*>::value == true;
//     bool check_two = objc::traits::is_object<objc::types::ID>::value == true;
//     bool check_three = objc::traits::is_object<non_object*>::value == false;
//     CHECK(check_one);
//     CHECK(check_one_and_a_half);
//     CHECK(check_two);
//     CHECK(check_three);
// }

// TEST_CASE("[SFINAE] Confirm results of objc::traits::is_object<NSObject*>::value",
//           "[sfinae-objc-traits-confirm-NSObject-pointer-value]") {
//     bool check_one = objc::traits::detail::has_isa<objc_object>::value == true;
//     bool check_two = objc::traits::is_object<NSObject*>::value == true;
//     CHECK(check_one);
//     CHECK(check_two);
// }

// TEST_CASE("[SFINAE] Inspect the results from objc::traits::*ptr<T>::type and objc::traits::*ptr_t<T>",
//           "[sfinae-inspect-results-objc-traits-null-specifier-traits]") {
//
//     // im::Image image = im::Image();
//     using T = im::InterleavedImage<>*;
//     using terminator::nameof;
//
//     // T t = new im::InterleavedImage<>();
//     T t = new std::remove_pointer_t<T>();
//
//     objc::nullable_ptr<T>::type a1;
//     objc::nonnull_ptr<T>::type a2;
//     objc::unspecified_ptr<T>::type a3;
//     objc::nullable_ptr_t<T> a4;
//     objc::nonnull_ptr_t<T> a5;
//     objc::unspecified_ptr_t<T> a6;
//
//     WTF("Demangled typename from specifier traits:",
//
//         FF("\tT                                       = %s",  nameof(t)),
//
//         FF("\tobjc::traits::nullable_ptr<T>::type     = %s",  nameof(a1)),
//         FF("\tobjc::traits::nonnull_ptr<T>::type      = %s",  nameof(a2)),
//         FF("\tobjc::traits::unspecified_ptr<T>::type  = %s",  nameof(a3)),
//
//         FF("\tobjc::traits::nullable_ptr_t<T>         = %s",  nameof(a4)),
//         FF("\tobjc::traits::nonnull_ptr_t<T>          = %s",  nameof(a5)),
//         FF("\tobjc::traits::unspecified_ptr_t<T>      = %s",  nameof(a6))
//
//     );
//
//     delete t;
//
// }

