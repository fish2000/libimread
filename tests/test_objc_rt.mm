
#include <cstdlib>
#include <libimread/libimread.hpp>
#include <libimread/objc-rt.hh>

#include "include/catch.hpp"

namespace {
    
    TEST_CASE("[objc-rt] Send a message via objc::msg::send()", "[objc-rt-msg-send]") {
        NSData *datum;
        NSURL *url;
        std::size_t nbytes = 20 * 1024;
        unsigned char randos[nbytes] = {0};
        
        @autoreleasepool {
            arc4random_buf(static_cast<void*>(randos), nbytes);
            datum = [[NSData alloc] initWithBytes:(const void *)&randos[0]
                                           length:(NSInteger)nbytes];
            objc::msg::send(datum,
                objc::selector("writeToURL:atomically:"),
                url, YES);
        };
    }
    
}