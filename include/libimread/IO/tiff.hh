/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_TIFF_HH_
#define LIBIMREAD_IO_TIFF_HH_

#include <utility>
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/pixels.hh>

namespace im {
    
    namespace detail {
        
        inline decltype(auto) writeopts() {
            return D(
                _compress = true,
                _horizontal_predictor = false,
                _metadata = false,
                _software_signature = "libimread (OST-MLOBJ/747)",
                _x_resolution = 72,
                _y_resolution = 72,
                _resolution_unit = 1,   /// RESUNIT_INCH
                _orientation = 1        /// ORIENTATION_TOPLEFT
            );
        }
        
    }
    
    class STKFormat : public ImageFormatBase<STKFormat> {
        public:
            using can_read = std::true_type;
            using can_read_multi = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x49\x49\x2a\x00", 4)
                },
                _suffixes = { "stk", "tif", "tiff" },
                _mimetype = "image/stk",
                _metadata = "<TIFF/STK METADATA STRING>",
                _writeopts = detail::writeopts()
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts) override {
                ImageList pages = this->do_read(src, factory, false, opts);
                std::unique_ptr<Image> out = pages.pop();
                return out;
            }
            
            virtual ImageList read_multi(byte_source* src,
                                        ImageFactory* factory,
                                        options_map const& opts) override {
                return this->do_read(src, factory, true, opts);
            }
            
        private:
            ImageList do_read(byte_source* src,
                              ImageFactory* factory,
                              bool is_multi,
                              options_map const& opts);
        
    };
    
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
                    SIGNATURE("\x4d\x4d\x00\x2b", 4)
                },
                _suffixes = { "tif", "tiff" },
                _mimetype = "image/tiff",
                _metadata = "<TIFF METADATA STRING>",
                _writeopts = detail::writeopts()
            );
            
            static bool match_format(byte_source* src) {
                return match_magic(src, "\x4d\x4d\x00\x2a", 4) ||
                       match_magic(src, "\x4d\x4d\x00\x2b", 4);
            }
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts) override {
                ImageList pages = this->do_read(src, factory, false, opts);
                std::unique_ptr<Image> out = pages.pop();
                return out;
            }
            
            virtual ImageList read_multi(byte_source* src,
                                         ImageFactory* factory,
                                         options_map const& opts) override {
                if (opts.cast<std::string>("format", "tif") == "stk") {
                    std::unique_ptr<STKFormat> delegate = std::make_unique<STKFormat>();
                    return delegate->read_multi(src, factory, opts);
                }
                return this->do_read(src, factory, true, opts);
            }
            
            virtual void write(Image& input,
                               byte_sink* output,
                               options_map const& opts) override;
            
        private:
            ImageList do_read(byte_source* src,
                              ImageFactory* factory,
                              bool is_multi,
                              options_map const& opts);
    };
    
    namespace format {
        using STK = STKFormat;
        using TIFF = TIFFFormat;
    }
    
}



#endif /// LIBIMREAD_IO_TIFF_HH_
