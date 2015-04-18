
#ifndef LPC_PPM_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012
#define LPC_PPM_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012

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

#endif /// LPC_PPM_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012

