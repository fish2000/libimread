// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_APPLE_HH_INCLUDE_GUARD_
#define LPC_APPLE_HH_INCLUDE_GUARD_
#ifdef __OBJC__

#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#include <memory>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace cf {
    //CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
}

namespace ns {
    
}

namespace im {
    
    class NSImageFormat : public ImageFormat {
        
        public:
            typedef std::true_type can_read;
            /*
            bool can_read() const override { return true; }
            bool can_write() const override { return true; }
            */
            
            std::unique_ptr<Image> read(
                byte_source *src,
                ImageFactory *factory,
                const options_map &opts);
            
            void write(Image* input,
                byte_sink* output,
                const options_map& opts);
    };

}

#endif
#endif // LPC_APPLE_HH_INCLUDE_GUARD_