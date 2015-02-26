// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_BMP_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012
#define LPC_BMP_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012

#include <libimread/base.hh>
#include <libimread/tools.hh>
#include <sstream>

namespace im {

    class BMPFormat : public ImageFormat {
        public:
            bool can_read() const override { return true; }
            bool can_write() const override { return false; }

            std::unique_ptr<Image> read(byte_source* src, ImageFactory* factory, const options_map& opts) override;
    };

}

#endif // LPC_BMP_H_INCLUDE_GUARD_THU_OCT_25_20_16_30_WEST_2012

