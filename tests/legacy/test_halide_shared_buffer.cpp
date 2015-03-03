
//#include <stdint.h>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>

#include "include/catch.hpp"

TEST_CASE("Call new_shared_buffer()", "[new-shared-buffer]") {
    im::shared_buffer buf = im::new_shared_buffer(1024, 768, 3);
}

TEST_CASE("Multiple calls to new_shared_buffer()", "[new-shared-buffers]") {
    im::shared_buffer buf = im::new_shared_buffer(1024, 768, 3);
    im::shared_buffer buf2 = im::new_shared_buffer(1024, 768, 3);
}

TEST_CASE("Call make_shared_buffer()", "[make-shared-buffer]") {
    buffer_t b = {0};
    im::shared_buffer buf = im::make_shared_buffer(b, 1024, 768, 3);
}

TEST_CASE("Multiple calls to make_shared_buffer()", "[new-shared-buffers]") {
    buffer_t b = {0};
    im::shared_buffer buf = im::make_shared_buffer(b, 1024, 768, 3);
    im::shared_buffer buf2 = im::make_shared_buffer(b, 1024, 768, 3);
}

TEST_CASE("Check the dimensions of a new buffer", "[shared-buffer-dims]") {
    buffer_t b = {0};
    im::shared_buffer buf = im::make_shared_buffer(b, 1024, 768, 3);
    
    WARN( "extent[0] = " << buf->extent[0] );
    
    CAPTURE( buf->extent[0] );
    CAPTURE( buf->extent[1] );
    CAPTURE( buf->extent[2] );
    CAPTURE( buf->extent[3] );

    CAPTURE( buf->stride[0] );
    CAPTURE( buf->stride[1] );
    CAPTURE( buf->stride[2] );
    CAPTURE( buf->stride[3] );
    
    REQUIRE( buf->extent[0] == 768 );
    REQUIRE( buf->extent[1] == 1024 );
    REQUIRE( buf->extent[2] == 3 );
    
    
}
