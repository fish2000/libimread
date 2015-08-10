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

    class TIFFFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            typedef std::true_type can_read_multi;
            typedef std::true_type can_read_metadata;
            typedef std::true_type can_write;
            
            static bool match_format(byte_source *src) {
                return match_magic(src, "\x4d\x4d\x00\x2a", 4) ||
                       match_magic(src, "\x4d\x4d\x00\x2b", 4) ||
                       match_magic(src, "\x49\x49\x2a\x00", 4);
            }
            
            virtual std::unique_ptr<Image> read(
                    byte_source *s,
                    ImageFactory *f,
                    const options_map &opts) override {
                std::unique_ptr<ImageList> pages = this->do_read(s, f, false);
                if (pages->size() != 1) { throw ProgrammingError(); }
                std::vector<Image*> ims = pages->release();
                return std::unique_ptr<Image>(ims[0]);
            }
            
            virtual std::unique_ptr<ImageList> read_multi(
                    byte_source *s,
                    ImageFactory *f,
                    const options_map &opts) override {
                return this->do_read(s, f, true);
            }
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
            
        private:
            std::unique_ptr<ImageList> do_read(byte_source *s,
                                                ImageFactory *f,
                                                bool is_multi);
    };
    
    class STKFormat : public ImageFormat {
        public:
            typedef std::true_type can_read_multi;
            
            virtual std::unique_ptr<ImageList> read_multi(
                byte_source *s,
                ImageFactory *f,
                const options_map &opts) override;
    };
    
    namespace format {
        using TIFF = TIFFFormat;
        using STK = STKFormat;
    }
    
}



#endif /// LIBIMREAD_IO_TIFF_HH_
