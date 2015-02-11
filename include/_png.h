// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
#define LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012

namespace im {

    class PNGFormat : public ImageFormat {
        public:
            bool can_read() const override { return true; }
            bool can_write() const override { return true; }
            
            /// NOT AN OVERRIDE:
            static bool match_format(byte_source* src) { return match_magic(src, "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A", 8); }

            std::auto_ptr<Image> read(byte_source* src, ImageFactory* factory, const options_map& opts) override;
            void write(Image* input, byte_sink* output, const options_map& opts) override;
    };


}


#endif // LPC_PNG_H_INCLUDE_GUARD_WED_FEB__1_16_34_50_WET_2012
