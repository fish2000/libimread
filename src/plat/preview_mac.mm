/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/preview.hh>
#include <chrono>
#include <thread>
#import <libimread/ext/categories/NSURL+IM.hh>

namespace im {
    
    namespace image {
        
        void preview(path const& timage) {
            @autoreleasepool {
                /// show
                NSURL* showurl = [NSURL fileURLWithFilesystemPath:timage];
                [showurl preview];
                /// sleep
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(
                        im::image::sleeptime));
            }
        }
        
    }
    
}

