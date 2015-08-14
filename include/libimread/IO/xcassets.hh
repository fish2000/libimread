/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_XCASSETS_HH_
#define LIBIMREAD_IO_XCASSETS_HH_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/IO/png.hh>

namespace im {
    
    class XCAssetsFormat : public PNGFormat {
        public:
            using can_write = PNGFormat::can_write;
            
            static bool match_format(byte_source *src) {
                /// not sure how this will actually work,
                /// since this "format" is actually
                /// a folder full of PNGs; not something
                /// easily ready in as a byte_source...
                return PNGFormat::match_format(src);
            }
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
    };
    
    namespace format {
        using XCAssets = XCAssetsFormat;
    }
    
}


#endif /// LIBIMREAD_IO_XCASSETS_HH_