/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_HH_
#define LIBIMREAD_COREGRAPHICS_HH_

#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#include <libimread/objc-rt.hh>
#endif

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/IO/apple.hh>

namespace im {
    
    using MetaImage = ImageWithMetadata;
    
    
    class CoreGraphicsImage : public Image, public MetaImage {
        
        
        
    };
    
    namespace apple {
        
        //using halide::default_opts;
        static const options_map apple_default_opts;
        
        template <typename T>
        using ImageType = HybridImage<typename std::decay<T>::type>;
        template <typename T>
        using image_ptr = std::unique_ptr<ImageType<T>>;
        
        /*
        namespace detail {
            /// XXX: remind me why in fuck did I write this shit originally
            template <typename T, typename pT>
            std::unique_ptr<T> dynamic_cast_unique(std::unique_ptr<pT> &&src) {
                /// Force a dynamic_cast upon a unique_ptr via interim swap
                /// ... danger, will robinson: DELETERS/ALLOCATORS NOT WELCOME
                /// ... from http://stackoverflow.com/a/14777419/298171
                if (!src) { return std::unique_ptr<T>(); }
                
                /// Throws a std::bad_cast() if this doesn't work out
                T *dst = &dynamic_cast<T&>(*src.get());
                
                src.release();
                std::unique_ptr<T> ret(dst);
                return ret;
            }
        }
        
        template <typename T = byte>
        image_ptr<T> read_unique(const std::string &filename,
                                 const options_map &opts = apple_default_opts) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            image_ptr<T> output = detail::dynamic_cast_unique<ImageType<T>>(
                    format->read(input.get(), &factory, opts));
            return output;
        }
        */
        
        template <typename T = byte> inline
        ImageType<T> read(const std::string &filename,
                          const options_map &opts = apple_default_opts) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            ImageType<T> image(dynamic_cast<ImageType<T>&>(*output));
            return image;
        }
        
    }
    
}

#endif /// LIBIMREAD_COREGRAPHICS_HH_