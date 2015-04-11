// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_APPLE_HH_INCLUDE_GUARD_
#define LPC_APPLE_HH_INCLUDE_GUARD_

#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#endif

#include <memory>
#include <cstdio>
#include <cstring>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    #ifdef __OBJC__
    namespace cf {
        
        // using Releaser = std::function<void(CFTypeRef)>;
        // Releaser basic =      [](CFTypeRef cf)        { CFRelease(cf); };
        // Releaser image =      [](CGImageRef cg)       { CFRelease(cf); };
        // Releaser context =    [](CGContextRef cgc)    { CGContextRelease(cgc); };
        
        template <typename C = CFTypeRef>
        struct unref {
            constexpr unref() noexcept = default;
            template <typename U> unref(const unref<U>&) noexcept {};
            void operator()(C ref) { if (ref) CFRelease(ref); }
        };
        
        template <typename Ref>
        using REF = std::unique_ptr<typename std::decay<Ref>::type, unref<Ref>>;
        
        /// cf::REF<CGColorSpaceRef> colorSpace(CGColorSpaceCreateDeviceRGB());
        /// cf::REF<CGContextRef> ctx(CGBitmapContextCreate((NULL, width * imageScale, height * imageScale,
        ///     8, 0, colorSpace, kCGImageAlphaPremultipliedLast))
        /// cf::REF<CGImageRef> image(CGBitmapContextCreateImage(ctx));
        
    }
    #endif
    
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