/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_PPM_HH_
#define LIBIMREAD_IO_PPM_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    class PPMFormat : public ImageFormatBase<PPMFormat> {
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("P6", 2)
                },
                _suffixes = { "ppm" },
                _mimetype = "image/x-portable-pixmap"
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts) override;
            
            virtual void write(Image& input,
                               byte_sink* output,
                               Options const& opts) override;
    };
    
    namespace format {
        using PPM = PPMFormat;
    }
    
}

#endif /// LIBIMREAD_IO_PNG_HH_