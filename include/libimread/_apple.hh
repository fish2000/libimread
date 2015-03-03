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
    
    template <typename CF>
    struct deleter {
        constexpr deleter() noexcept = default;
        template <class U>
        deleter(const deleter<U>&) noexcept {};
        void operator()(CF_CONSUMED CF obj) { if (obj) CFRelease(obj); }
    };
    
    template<typename CF>
    using ref = std::unique_ptr<typename std::remove_pointer<CF>::type, deleter<CF>>;
    
    /*
    template <typename CF = CFType>
    struct ref : public std::enable_shared_from_this<ref> {
        CF r;
        
        ref() {}
        ref(CF &r_) : r(r_) {}
        
        std::shared_ptr<ref> shared() { return shared_from_this(); }
        // inline r *get() { return shared().get(); }
        // inline r *operator->() { return get(); }
        // inline r operator*() { return *get(); }
        inline bool operator==(const CF &other) { return (bool)CFEqual(r, other); }
        int retain() { CFRetain(r); return (int)CFGetRetainCount(r); }
        void release() { CFRelease(r); }
    };
    */
    
    
    
    
}

namespace im {
    
    class PVRTCFormat : public ImageFormat {
        public:
            bool can_read() const override { return true; }
            bool can_write() const override { return false; }
            std::unique_ptr<Image> read(byte_source *src, ImageFactory *factory, const options_map &opts) override;
    };

}

#endif
#endif // LPC_APPLE_HH_INCLUDE_GUARD_