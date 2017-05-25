// Copyright 2017 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#include <memory>
#include <cstdlib>
#include <libimread/IO/gif.hh>

#import  <CoreFoundation/CoreFoundation.h>
// #import  <Foundation/Foundation.h>
#import  <ImageIO/ImageIO.h>

namespace im {
    
    namespace detail {
        
        using pixbuf_t = std::unique_ptr<uint32_t[]>;
        
        template <typename CF>
        struct cfreleaser {
            constexpr cfreleaser() noexcept = default;
            template <typename U> cfreleaser(cfreleaser<U> const&) noexcept {};
            void operator()(CF cfp) { CFRelease(cfp); cfp = nil; }
        };
        
        template <typename CoreFoundationType>
        using cfp_t = std::unique_ptr<
                      typename std::decay_t<
                               std::remove_pointer_t<CoreFoundationType>>,
                                          cfreleaser<CoreFoundationType>>;
        
        // template <typename CoreFoundationType>
        // using cfp_t = std::unique_ptr<
        //               std::remove_pointer_t<CoreFoundationType>,
        //               decltype(CFRelease)&>
        // #define CF_SCOPED(name, instance) detail::cfp name{ instance, CFRelease }
        
    }
    
    std::unique_ptr<Image> GIFFormat::read(byte_source* src,
                                           ImageFactory* factory,
                                           options_map const& opts) {
        /// YO DOGG
        char const* yo = "DOGG";
        
        /// CFURL
        
        
        /// CFData
        
        
        /// CGImageRef
        // CGImageRef imageref = nullptr;
        detail::cfp_t<CGImageRef> imageref(nullptr);
        
        // CGColorSpaceRef csref = CGColorSpaceCreateDeviceRGB();
        // CF_SCOPED(cfref, CGColorSpaceCreateDeviceRGB());
        
        detail::cfp_t<CGColorSpaceRef> colorspace(CGColorSpaceCreateDeviceRGB());
        
        std::size_t width = CGImageGetWidth(imageref.get()),
                   height = CGImageGetHeight(imageref.get()),
                      bpp = 4,                              /// bytes per pixel
                      bpr = bpp * width,                    /// bytes per row
                      bpc = 8;                              /// bits per component
        
        // uint32_t* pixelbuffer;
        detail::pixbuf_t pixbuf = std::make_unique<uint32_t[]>(width * height);
        
        /// CGContextRef
        detail::cfp_t<CGContextRef> context(
                      CGBitmapContextCreate(pixbuf.get(),
                                            width, height,
                                            bpc, bpr, colorspace.get(),
                                            kCGImageAlphaPremultipliedLast |
                                            kCGBitmapByteOrder32Big));
        
        CGContextDrawImage(context.get(),
                           CGRectMake(0, 0, width, height),
                           imageref.get());
        
        /// pixel loop
        return nullptr;
    }
    
}