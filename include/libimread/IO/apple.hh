/// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_APPLE_HH_
#define LIBIMREAD_IO_APPLE_HH_

#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
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

#endif /// LIBIMREAD_IO_APPLE_HH_