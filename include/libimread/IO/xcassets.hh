// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_XCASSETS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_XCASSETS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/IO/png.hh>

namespace im {
    
    class XCAssetsFormat : public PNGFormat {
        public:
            typedef PNGFormat::can_write    can_write;
            
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


#endif // LPC_XCASSETS_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
