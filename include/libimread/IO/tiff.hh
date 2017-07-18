/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_TIFF_HH_
#define LIBIMREAD_IO_TIFF_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>
#include <libimread/imagelist.hh>

namespace im {
    
    namespace detail {
        
        inline decltype(auto) readopts() {
            return D(
                _samples_per_pixel = 1,
                _bits_per_sample = 8
            );
        }
        
        inline decltype(auto) writeopts() {
            return D(
                _compress = true,
                _copy_data = true,
                _horizontal_predictor = false,
                _metadata = false,
                _software_signature = "libimread (OST-MLOBJ/747)",
                _x_resolution = 72,
                _y_resolution = 72,
                _resolution_unit = 1,   /// RESUNIT_INCH
                _orientation = 1        /// ORIENTATION_TOPLEFT
            );
        }
        
    } /* namespace detail */
    
    class TIFFFormat : public ImageFormatBase<TIFFFormat> {
        
        public:
            
            using can_read = std::true_type;
            using can_read_multi = std::true_type;
            // using can_read_metadata = std::true_type;
            using can_write = std::true_type;
            using can_write_multi = std::true_type;
            // using can_write_metadata = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x4d\x4d\x00", 3),
                    SIGNATURE("\x4d\x4d\x00\x2a", 4),
                    SIGNATURE("\x4d\x4d\x00\x2b", 4)
                },
                _suffixes = { "tif", "tiff" },
                _mimetype = "image/tiff",
                _metadata = "<TIFF METADATA STRING>",
                _readopts = detail::readopts(),
                _writeopts = detail::writeopts()
            );
            
            /// Custom match_format() using the latter two byte signatures:
            static bool match_format(byte_source* src);
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts) override {
                ImageList pages = do_read(src, factory, false, opts);
                std::unique_ptr<Image> out = pages.pop();
                return out;
            }
            
            virtual ImageList read_multi(byte_source* src,
                                         ImageFactory* factory,
                                         options_map const& opts) override {
                return do_read(src, factory, true, opts);
            }
            
            virtual void write(Image& input,
                               byte_sink* output,
                               options_map const& opts) override;
            
            virtual void write_multi(ImageList& input,
                                     byte_sink* output,
                                     options_map const& opts) override;
            
        private:
            
            ImageList do_read(byte_source* src,
                              ImageFactory* factory,
                              bool is_multi,
                              options_map const& opts);
            
            void     do_write(ImageList& input,
                              byte_sink* output,
                              bool is_multi,
                              options_map const& opts);
    };
    
    class STKFormat : public ImageFormatBase<STKFormat> {
        
        public:
            
            using can_read = std::true_type;
            using can_read_multi = std::true_type;
            // using can_read_metadata = std::true_type;
            using can_write = std::true_type;
            using can_write_multi = std::true_type;
            // using can_write_metadata = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x49\x49\x2a\x00", 4)
                },
                _suffixes = { "stk", "tif", "tiff" },
                _mimetype = "image/stk",
                _metadata = "<TIFF/STK METADATA STRING>",
                _readopts = detail::readopts(),
                _writeopts = detail::writeopts()
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                options_map const& opts) override {
                ImageList pages = do_read(src, factory, false, opts);
                std::unique_ptr<Image> out = pages.pop();
                return out;
            }
            
            virtual ImageList read_multi(byte_source* src,
                                         ImageFactory* factory,
                                         options_map const& opts) override {
                return do_read(src, factory, true, opts);
            }
            
            virtual void write(Image& input,
                               byte_sink* output,
                               options_map const& opts) override {
                std::unique_ptr<TIFFFormat> delegate = std::make_unique<TIFFFormat>();
                return delegate->write(input, output, opts);
            }
            
            virtual void write_multi(ImageList& input,
                                     byte_sink* output,
                                     options_map const& opts) override {
                std::unique_ptr<TIFFFormat> delegate = std::make_unique<TIFFFormat>();
                return delegate->write_multi(input, output, opts);
            }
            
        private:
            
            ImageList do_read(byte_source* src,
                              ImageFactory* factory,
                              bool is_multi,
                              options_map const& opts);
        
    };
    
    namespace format {
        using TIFF = TIFFFormat;
        using TIF = TIFFFormat;
        using STK = STKFormat;
    }
    
} /* namespace im */


#endif /// LIBIMREAD_IO_TIFF_HH_