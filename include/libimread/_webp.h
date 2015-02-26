// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LPC_WEBP_H_INCLUDE_GUARD_
#define LPC_WEBP_H_INCLUDE_GUARD_

namespace im {

    class WebPFormat : public ImageFormat {
        public:
            bool can_read() const override { return true; }
            bool can_write() const override { return false; }

            std::unique_ptr<Image> read(byte_source* src, ImageFactory* factory, const options_map& opts) override;
    };

}


#endif // LPC_WEBP_H_INCLUDE_GUARD_
