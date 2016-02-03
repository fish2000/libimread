/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_HH_
#define LIBIMREAD_COREGRAPHICS_HH_

#ifdef __OBJC__
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#endif

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/halide.hh>
#include <libimread/interleaved.hh>
#include <libimread/IO/apple.hh>

namespace im {
    
    namespace apple {
        
        using filesystem::path;
        static const options_map apple_default_opts;
        
        template <typename T>
        using ImageType = HybridImage<typename std::decay_t<T>>;
        template <typename T>
        using image_ptr = std::unique_ptr<ImageType<T>>;
        
        template <typename T = byte> inline
        ImageType<T> read(const std::string &filename,
                          const options_map &opts = apple_default_opts) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            options_map default_opts = format->get_options();
            std::unique_ptr<Image> output = format->read(input.get(), &factory, default_opts.update(opts));
            ImageType<T> image(dynamic_cast<ImageType<T>&>(*output));
            return image;
        }
        
        template <typename T = byte> inline
        void write(HybridImage<T>& input, const std::string& filename,
                                          const options_map& opts = apple_default_opts) {
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            options_map writeopts(opts);
            writeopts.set("filename", filename);
            writeopts.set("extension", path::extension(filename));
            format->write(dynamic_cast<Image&>(input), output.get(), writeopts);
        }
        
        template <typename Color = color::RGBA> inline
        void write(InterleavedImage<Color>& input, const std::string& filename,
                                                   const options_map& opts = apple_default_opts) {
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            options_map writeopts(opts);
            writeopts.set("filename", filename);
            writeopts.set("extension", path::extension(filename));
            format->write(dynamic_cast<Image&>(input), output.get(), writeopts);
        }
        
    }
    
}

#endif /// LIBIMREAD_COREGRAPHICS_HH_