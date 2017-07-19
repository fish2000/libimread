/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_PNG_HH_
#define LIBIMREAD_IO_PNG_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    class PNGFormat : public ImageFormatBase<PNGFormat> {
        
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            /// left off the following bit from the signature:
            ///         \x0D\x0A\x1A\x0A
            /// ... due to complaints of a 'control character in std::string'
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x89\x50\x4E\x47\x0D\x0A\x1A\x0A", 8)
                },
                _suffixes = { "png" },
                _mimetype = "image/png",
                _standard_gamma = 0.45455,
                _primary_cromacities = D(
                    _white = D(_xx = 0.312700,
                               _yy = 0.329000),
                    _red   = D(_xx = 0.640000,
                               _yy = 0.330000),
                    _blue  = D(_xx = 0.300000,
                               _yy = 0.600000),
                    _green = D(_xx = 0.150000,
                               _yy = 0.060000))
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts) override;
            virtual void write(Image& input,
                               byte_sink* output,
                               Options const& opts) override;
            
            /// NOT AN OVERRIDE:
            virtual void write_ios(Image& input,
                                   byte_sink* output,
                                   Options const& opts);
    };
    
    namespace format {
        using PNG = PNGFormat;
    }
    
}


#endif /// LIBIMREAD_IO_PNG_HH_
