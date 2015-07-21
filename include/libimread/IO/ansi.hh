/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_ANSI_HH_
#define LIBIMREAD_IO_ANSI_HH_

#include <cstddef>
#include <cwctype>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/base.hh>

namespace im {
    
    namespace detail {
        
    }
    
    class ANSIFormat : public ImageFormat {
        public:
            typedef std::true_type can_write;
            typedef std::true_type can_write_multi;
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
            
            virtual void write_multi(std::vector<Image> &input,
                                     byte_sink* output,
                                     const options_map &opts) override;
    };
    
    namespace format {
        using ANSI = ANSIFormat;
    }
    
}

#endif /// LIBIMREAD_IO_ANSI_HH_
