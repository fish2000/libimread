// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012
#define LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012

#include "base.h"

namespace im {

    class JPEGFormat : public ImageFormat {
        public:
            bool can_read() const override { return true; }
            bool can_write() const override { return true; }

            /// NOT AN OVERRIDE:
            static bool match_format(byte_source* src) { return match_magic(src, "\xff\xd8\xff", 3); }

            std::unique_ptr<Image> read(byte_source* src, ImageFactory* factory, const options_map& opts) override;
            void write(Image* input, byte_sink* output, const options_map& opts) override;
    };

}


#endif // LPC_JPEG_H_INCLUDE_GUARD_THU_FEB__2_18_14_07_WET_2012
