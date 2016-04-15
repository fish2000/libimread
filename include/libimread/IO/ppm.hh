/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_PPM_HH_
#define LIBIMREAD_IO_PPM_HH_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class PPMFormat : public ImageFormatBase<PPMFormat> {
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            DECLARE_OPTIONS(
                base64::encode("P6", 2),
                "ppm",
                "image/x-portable-pixmap");
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                const options_map& opts) override;
            
            virtual void write(Image& input,
                               byte_sink* output,
                               const options_map& opts) override;
    };
    
    namespace format {
        using PPM = PPMFormat;
    }
    
}

#endif /// LIBIMREAD_IO_PNG_HH_