/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_TIFF_HH_
#define LIBIMREAD_IO_TIFF_HH_

#include <utility>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cstring>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/pixels.hh>

namespace im {

    class TIFFFormat : public ImageFormatBase<TIFFFormat> {
        public:
            using can_read = std::true_type;
            using can_read_multi = std::true_type;
            using can_read_metadata = std::true_type;
            using can_write = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x4d\x4d\x00", 3),
                    SIGNATURE("\x4d\x4d\x00\x2a", 4),
                    SIGNATURE("\x4d\x4d\x00\x2b", 4),
                    SIGNATURE("\x49\x49\x2a\x00", 4)
                },
                _suffixes = { "tif", "tiff" },
                _mimetype = "image/tiff",
                _metadata = "<TIFF METADATA STRING>",
                _writeopts = D(
                    _compress = true,
                    _horizontal_predictor = false,
                    _metadata = false,
                    _software_signature = "libimread (OST-MLOBJ/747)",
                    _x_resolution = 72,
                    _y_resolution = 72,
                    _resolution_unit = 1,   /// RESUNIT_INCH
                    _orientation = 1        /// ORIENTATION_TOPLEFT
                )
            );
            
            static bool match_format(byte_source* src) {
                return match_magic(src, "\x4d\x4d\x00\x2a", 4) ||
                       match_magic(src, "\x4d\x4d\x00\x2b", 4) ||
                       match_magic(src, "\x49\x49\x2a\x00", 4);
            }
            
            virtual std::unique_ptr<Image> read(byte_source* s,
                                                ImageFactory* f,
                                                const options_map &opts) override {
                ImageList pages = this->do_read(s, f, false);
                if (pages.size() != 1) { throw ProgrammingError(); }
                return pages.pop();
            }
            
            virtual ImageList read_multi(byte_source* s,
                                         ImageFactory* f,
                                         const options_map& opts) override {
                return this->do_read(s, f, true);
            }
            
            virtual void write(Image& input,
                               byte_sink* output,
                               const options_map& opts) override;
            
        private:
            ImageList do_read(byte_source* s,
                              ImageFactory* f,
                              bool is_multi);
    };
    
    class STKFormat : public ImageFormatBase<STKFormat> {
        public:
            using can_read_multi = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    D(_bytes    = base64::encode("\x49\x49\x2a\x00", 4),
                      _length   = 4)
                },
                _suffixes = { "stk" },
                _mimetype = "image/stk"
            );
            
            virtual ImageList read_multi(byte_source* s,
                                         ImageFactory* f,
                                         const options_map &opts) override;
    };
    
    namespace format {
        using TIFF = TIFFFormat;
        using STK = STKFormat;
    }
    
}



#endif /// LIBIMREAD_IO_TIFF_HH_
