// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_APPLE_HH_INCLUDE_GUARD_
#define LPC_APPLE_HH_INCLUDE_GUARD_

#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#import <libimread/ext/UTI.h>
#endif

#include <memory>
#include <cstdio>
#include <cstring>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class NSImageFormat : public ImageFormat {
        
        public:
            typedef std::true_type can_read;
            
            std::unique_ptr<Image> read(
                byte_source *src,
                ImageFactory *factory,
                const options_map &opts);
    };
    
    namespace format {
        using NS = NSImageFormat;
        using Apple = NSImageFormat;
    }

}

#endif // LPC_APPLE_HH_INCLUDE_GUARD_