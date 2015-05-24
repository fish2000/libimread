
#ifndef LIBIMREAD_IO_PPM_HH_
#define LIBIMREAD_IO_PPM_HH_

#include <libimread/libimread.hpp>
#include <libimread/base.hh>

namespace im {

    class PPMFormat : public ImageFormat {
        public:
            typedef std::true_type can_read;
            typedef std::true_type can_write;
            
            virtual std::unique_ptr<Image> read(byte_source *src,
                                                ImageFactory *factory,
                                                const options_map &opts) override;
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
    };
    
    namespace format {
        using PPM = PPMFormat;
    }
    
}

#endif /// LIBIMREAD_IO_PNG_HH_