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
            using can_write = std::true_type;
            
        public:
            /// PVRTC "magic" tag is internal:
            DECLARE_OPTIONS(
                _signatures = { 0x21525650 },
                _suffixes = { "pvr", "pvrtc" },
                _mimetype = "image/x-pvr"
            );
            
        public:
            /// bespoke version of “match_format()” --
            /// reads just enough bytes to construct a PVRHeader,
            /// and then inspect the relevant “magic” member value
            static bool match_format(byte_source* src);
            
            /// PVRTC read -- makes use of:
            ///   * libimread/include/ext/pvrtc.hh
            ///   * libimread/src/ext/pvr.cpp
            ///   * libimread/src/ext/pvrtc.cpp
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts) override;
            
            /// PVRTC write -- makes use of:
            ///   * libimread/include/endian.hh
            ///   * libimread/include/imageref.hh
            ///   * libimread/include/image.hh
            ///   * libimread/src/image.cpp
            ///   * libimread/include/ext/boundaries.hh
            ///   * libimread/include/crop.hh
            ///   * deps/imagecompression/image_compression/internal/color_types.h
            ///   * deps/imagecompression/image_compression/internal/color_util.h
            ///   * deps/imagecompression/image_compression/public/compressed_image.h
            ///   * deps/imagecompression/image_compression/public/pvrtc_compressor.h
            ///   * deps/imagecompression/image_compression/internal/pvrtc_compressor.cc
            virtual void write(Image& input,
                               byte_sink* output,
                               Options const& opts) override;

    };
    
    namespace format {
        using PVR = PVRTCFormat;
        using PVRTC = PVRTCFormat;
    }
    
}

#endif /// LIBIMREAD_IO_PVRTC_HH_
