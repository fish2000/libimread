/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_HH_
#define LIBIMREAD_COREGRAPHICS_HH_

#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#include <libimread/objc-rt/objc-rt.hh>
#endif

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/IO/apple.hh>

namespace im {
    
    /*
    using MetaImage = ImageWithMetadata;
    class CoreGraphicsImage : public Image, public MetaImage {};
    */
    
    namespace apple {
        
        //using halide::default_opts;
        static const options_map apple_default_opts;
        
        template <typename T>
        using ImageType = HybridImage<typename std::decay<T>::type>;
        template <typename T>
        using image_ptr = std::unique_ptr<ImageType<T>>;
        
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