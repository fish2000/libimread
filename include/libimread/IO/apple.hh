/// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_APPLE_HH_
#define LIBIMREAD_IO_APPLE_HH_

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/base.hh>

namespace im {
    
    // namespace detail {
    //
    //     __attribute__((ns_returns_retained))
    //     NSDictionary* translate_options(options_map const& opts);
    //
    // }
    
    class NSImageFormat : public ImageFormatBase<NSImageFormat> {
        
        public:
            using can_read = std::true_type;
            using can_write = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("AppleImage", 10)
                },
                _suffixes = { "ns", "objc" },
                _mimetype = "x-image/apple"
            );
            
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