/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_JPEG_HH_
#define LIBIMREAD_IO_JPEG_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    class JPEGFormat : public ImageFormatBase<JPEGFormat> {
        
        public:
            using can_read = std::true_type;
            using can_read_metadata = std::true_type;
            using can_write = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\xff\xd8\xff", 3)
                },
                _suffixes = { "jpg", "jpeg", "jpe" },
                _mimetype = "image/jpeg",
                _buffer_size = 4096,
                _quantization = 256,
                _readopts = D(
                    _read_icc_profile = false
                ),
                _writeopts = D(
                    _progressive = false,
                    _quality = 0.75
                )
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts) override;
            
            virtual Metadata read_metadata(byte_source* src,
                                           Options const& opts) override;
            
            virtual void write(Image& input,
                               byte_sink* output,
                               Options const& opts) override;
    };
    
    namespace format {
        using JPG = JPEGFormat;
        using JPEG = JPEGFormat;
    }
    
}


#endif /// LIBIMREAD_IO_JPEG_HH_
