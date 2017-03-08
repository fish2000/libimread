/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_PVRTC_HH_
#define LIBIMREAD_IO_PVRTC_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>

namespace im {
    
    class PVRTCFormat : public ImageFormatBase<PVRTCFormat> {
        public:
            using can_read = std::true_type;
            
            /// PVRTC "magic" tag is internal
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("4\x00\x00\x00", 4)
                },
                _suffixes = { "pvr", "pvrtc" },
                _mimetype = "image/x-pvr"
            );
            
            static bool match_format(byte_source* src);
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts) override;
    };
    
    namespace format {
        using PVR = PVRTCFormat;
        using PVRTC = PVRTCFormat;
    }
    

}


#endif /// LIBIMREAD_IO_PVRTC_HH_
