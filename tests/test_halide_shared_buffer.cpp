
#include <stdint.h>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>

#include "include/catch.hpp"

TEST_CASE("imread|halide|shared-buffer-new") {
    im::shared_buffer buf = im::new_shared_buffer<uint8_t, buffer_t>(1024, 768, 3);
}

TEST_CASE("imread|halide|shared-buffers-new") {
    im::shared_buffer buf = im::new_shared_buffer<uint8_t, buffer_t>(1024, 768, 3);
    im::shared_buffer buf2 = im::new_shared_buffer<uint8_t, buffer_t>(1024, 768, 3);
}

TEST_CASE("imread|halide|shared-buffer-dims") {
    im::shared_buffer buf = im::new_shared_buffer<uint8_t, buffer_t>(1024, 768, 3);
    REQUIRE( buf->extent[0] == 768 );
    REQUIRE( buf->extent[1] == 1024 );
    REQUIRE( buf->extent[2] == 3 );
}
