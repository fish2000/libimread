/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/preview.hh>
#include <chrono>
#include <thread>

// #import <Foundation/Foundation.h>

namespace im {
    
    namespace image {
        
        void preview(path const& timage) {
            /// launch with Preview
            // NSString* filePath = [[NSString alloc] initWithUTF8String:timage.make_absolute().c_str()];
            // BOOL launched =  [[NSWorkspace sharedWorkspace] openFile:filePath
            //                                          withApplication:@"Preview.app"];
            /// sleep
            std::this_thread::sleep_for(
                std::chrono::milliseconds(
                    im::image::sleeptime));
        }
        
    }
    
}

