// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
#define LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {

    class TIFFFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            typedef std::true_type can_read_multi;
            typedef std::true_type can_write;
            typedef std::true_type can_read_metadata;
            
            std::unique_ptr<Image> read(byte_source *s,
                                        ImageFactory *f,
                                        const options_map &opts)  {
                std::unique_ptr<image_list> pages = this->do_read(s, f, false);
                if (pages->size() != 1) { throw ProgrammingError(); }
                std::vector<Image*> ims = pages->release();
                return std::unique_ptr<Image>(ims[0]);
            }
            
            std::unique_ptr<image_list> read_multi(byte_source *s,
                                                   ImageFactory *f,
                                                   const options_map &opts)  {
                return this->do_read(s, f, true);
            }
            
            void write(Image *input,
                       byte_sink *output,
                       const options_map &opts);
            
        private:
            std::unique_ptr<image_list> do_read(byte_source *s,
                                                ImageFactory *f,
                                                bool is_multi);
    };
    
    class STKFormat : public ImageFormat {
        public:
            typedef std::true_type can_read_multi;
            
            std::unique_ptr<image_list> read_multi(byte_source *s,
                                                   ImageFactory *f,
                                                   const options_map &opts);
    };
    
    namespace format {
        using TIFF = TIFFFormat;
        using STK = STKFormat;
    }
    
}



#endif // LPC_TIFF_INCLUDE_GUARD_Wed_Feb__8_19_02_16_WET_2012
