// Copyright 2014 Alexander Böhn <fish2000@gmail.com>.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_BMP_HH_
#define LIBIMREAD_IO_BMP_HH_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {
    
    class BMPFormat : public ImageFormatBase<BMPFormat> {
        public:
            using can_read = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x42\x4d", 2)
                },
                _suffixes = { "bmp" },
                _mimetype = "image/x-bmp"
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts) override;
    };
    
    namespace format {
        using BMP = BMPFormat;
    }
    
}

#endif /// LIBIMREAD_IO_BMP_HH_
