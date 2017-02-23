
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

namespace {
    
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
    
} /// namespace (anon.)