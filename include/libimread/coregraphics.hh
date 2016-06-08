/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_HH_
#define LIBIMREAD_COREGRAPHICS_HH_

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/halide.hh>
#include <libimread/interleaved.hh>
#include <libimread/IO/apple.hh>

namespace im {
    
    namespace apple {
        
        using filesystem::path;
        
        template <typename T>
        using ImageType = HybridImage<std::decay_t<T>>;
        
        template <typename T>
        using image_ptr = std::unique_ptr<ImageType<T>>;
        
        template <typename T = byte> inline
        ImageType<T> read(std::string const& filename,
                          options_map const& opts = options_map()) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(new format::Apple());
            std::unique_ptr<FileSource> input(new FileSource(filename));
            options_map readopts(format->add_options(opts));
            readopts.set("filename", filename);
            readopts.set("extension", path::extension(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, readopts);
            ImageType<T> image(dynamic_cast<ImageType<T>&>(*output));
            return image;
        }
        
        template <typename T = byte> inline
        void write(HybridImage<T>& input, std::string const& filename,
                                          options_map const& opts = options_map()) {
            std::unique_ptr<ImageFormat> format(new format::Apple());
            std::unique_ptr<FileSink> output(new FileSink(filename));
            options_map writeopts(format->add_options(opts));
            writeopts.set("filename", filename);
            writeopts.set("extension", path::extension(filename));
            format->write(dynamic_cast<Image&>(input), output.get(), writeopts);
        }
        
        template <typename Color = color::RGBA> inline
        void write(InterleavedImage<Color>& input, std::string const& filename,
                                                   options_map const& opts = options_map()) {
            std::unique_ptr<ImageFormat> format(new format::Apple());
            std::unique_ptr<FileSink> output(new FileSink(filename));
            options_map writeopts(format->add_options(opts));
            writeopts.set("filename", filename);
            writeopts.set("extension", path::extension(filename));
            format->write(dynamic_cast<Image&>(input), output.get(), writeopts);
        }
        
    }
    
}

#endif /// LIBIMREAD_COREGRAPHICS_HH_