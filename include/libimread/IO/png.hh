/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_PNG_HH_
#define LIBIMREAD_IO_PNG_HH_

#include <cstring>
#include <csetjmp>
#include <vector>
#include <sstream>

#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/options.hh>

namespace im {
    
    class PNGFormat : public ImageFormat {
        
        public:
            typedef std::true_type can_read;
            typedef std::true_type can_write;
            
            static const options_type OPTS() {
                const options_type O(
                    "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A",
                    "png"
                );
                return O;
            }
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
            
            /// ALSO NOT AN OVERRIDE:
            virtual void write_ios(Image &input,
                                   byte_sink *output,
                                   const options_map &opts);
    };
    
    namespace format {
        using PNG = PNGFormat;
    }
    
}


#endif /// LIBIMREAD_IO_PNG_HH_
