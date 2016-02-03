// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_APPLE_HH_INCLUDE_GUARD_
#define LPC_APPLE_HH_INCLUDE_GUARD_

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#endif /// __OBJC__

#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

#ifndef __OBJC__

/// forward-declare CFTypeRef type
struct CFTypeRef;

/// forward-declare CFRelease()
void CFRelease(CFTypeRef ref);

#endif

namespace im {
    
    namespace cf {
        
        template <typename C = CFTypeRef>
        struct unref {
            constexpr unref() noexcept = default;
            template <typename U> unref(const unref<U>&) noexcept {};
            void operator()(C ref) { if (ref) { CFRelease(ref); } }
        };
        
        template <typename Ref>
        using REF = std::unique_ptr<typename std::decay<Ref>::type, unref<Ref>>;
        
    }
    
    class NSImageFormat : public ImageFormatBase<NSImageFormat> {
        
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts) override;
            
            virtual void write(Image& input,
                               byte_sink* output,
                               const options_map& opts) override;
    };
    
    namespace format {
        using NS = NSImageFormat;
        using Apple = NSImageFormat;
    }

}

#endif // LPC_APPLE_HH_INCLUDE_GUARD_