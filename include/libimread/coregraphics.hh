// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_H_
#define LIBIMREAD_COREGRAPHICS_H_

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>

namespace im {
    
    namespace apple {
        
        static const options_map opts; /// NOT IN USE
        
        template <typename T>
        using ImageType = HybridImage<typename std::decay<T>::type>;
        template <typename T>
        using image_ptr = std::unique_ptr<ImageType<T>>;
        
        template <typename T = byte>
        image_ptr<T> read_unique(const std::string &filename) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            image_ptr<T> output = dynamic_cast_unique<ImageType<T>>(
                    format->read(input.get(), &factory, opts));
            return output;
        }
        
        template <typename T = byte>
        ImageType<T> read(const std::string &filename) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            ImageType<T> image(dynamic_cast<ImageType<T>&>(*output));
            return image;
        }
        
    }
    
}

#endif // LIBIMREAD_COREGRAPHICS_H_