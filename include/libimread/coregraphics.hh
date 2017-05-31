/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_HH_
#define LIBIMREAD_COREGRAPHICS_HH_

#include <memory>
#include <cstdlib>
#include <libimread/store.hh>

#import  <CoreFoundation/CoreFoundation.h>
#import  <ImageIO/ImageIO.h>

/// N.B. this next whole conditional import is just to define `kUTTypeGIF`
#if defined( __IPHONE_OS_VERSION_MIN_REQUIRED ) && __IPHONE_OS_VERSION_MIN_REQUIRED
#import <MobileCoreServices/MobileCoreServices.h>
#else
#import <CoreServices/CoreServices.h>
#endif

/// These macros, like much other stuff in this implementation TU,
/// are adapted from the always-handy-dandy Ray Wenderlich -- specifically, q.v.:
/// https://www.raywenderlich.com/69855/image-processing-in-ios-part-1-raw-bitmap-modification
/// http://www.modejong.com/blog/post3_pixel_binary_layout_w_premultiplied_alpha/index.html

#define UNCOMPAND(x)    ((x) & 0xFF)
#define R(x)            (UNCOMPAND(x))
#define G(x)            (UNCOMPAND(x >> 8 ))
#define B(x)            (UNCOMPAND(x >> 16))
#define A(x)            (UNCOMPAND(x >> 24))

#define CF_IDX(x)       static_cast<CFIndex>(x)
#define CG_FLOAT(x)     static_cast<CGFloat>(x)

CF_EXPORT
__attribute__((cf_returns_retained))
CFStringRef CFStringCreateWithSTLString(CFAllocatorRef alloc,
                                        char const* cStr,
                                        CFStringEncoding encoding = kCFStringEncodingUTF8);

CF_EXPORT
__attribute__((cf_returns_retained))
CFStringRef CFStringCreateWithSTLString(CFAllocatorRef alloc,
                                        std::string const& stlStr,
                                        CFStringEncoding encoding = kCFStringEncodingUTF8);

CF_EXPORT
std::string CFStringGetSTLString(CFStringRef theString,
                                 CFStringEncoding encoding = kCFStringEncodingUTF8);

namespace im {
    
    namespace detail {
        
        using pixbuf_t = std::unique_ptr<uint32_t[]>;
        
        template <typename CoreFoundationType>
        struct cfreleaser {
            constexpr cfreleaser() noexcept = default;
            template <typename U> cfreleaser(cfreleaser<U> const&) noexcept {};
            void operator()(CoreFoundationType __attribute__((cf_consumed)) cfp) {
                if (cfp) { CFRelease(cfp); cfp = nullptr; }
            }
        };
        
        template <typename CoreFoundationType>
        using cfp_t = std::unique_ptr<
                      typename std::decay_t<
                               std::remove_pointer_t<CoreFoundationType>>,
                                          cfreleaser<CoreFoundationType>>;
    }
    
}

namespace store {
    
    using cfmdict_ptr = im::detail::cfp_t<CFMutableDictionaryRef>;
    
    class cfdict : public store::stringmapper {
        
        public:
            DECLARE_STRINGMAPPER_TEMPLATES(cfdict);
        
        public:
            virtual bool can_store() const noexcept override;
        
        public:
            cfdict(void);
            cfdict(cfdict const&);
            cfdict(cfdict&&) noexcept;
            explicit cfdict(CFDictionaryRef);
            explicit cfdict(CFMutableDictionaryRef);
            virtual ~cfdict();
        
        protected:
            bool has(std::string const&) const;
            bool has(CFStringRef) const;
        
        public:
            std::string const& get_force(std::string const&) const;
        
        public:
            /// implementation of the store::stringmapper API,
            /// in terms of the CoreFoundation CFDictionary API
            virtual std::string&       get(std::string const& key) override;
            virtual std::string const& get(std::string const& key) const override;
            virtual bool set(std::string const& key, std::string const& value) override;
            virtual bool del(std::string const& key) override;
            virtual std::size_t count() const override;
            virtual stringvec_t list() const override;
        
        public:
            __attribute__((cf_returns_retained)) CFDictionaryRef dictionary() const;
            __attribute__((cf_returns_retained)) operator CFDictionaryRef() const;
            __attribute__((cf_returns_retained)) CFMutableDictionaryRef mutabledictionary() const;
            __attribute__((cf_returns_retained)) operator CFMutableDictionaryRef() const;
        
        protected:
            mutable cfmdict_ptr instance{ nullptr };
    };
    
}

#endif /// LIBIMREAD_COREGRAPHICS_HH_