// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#include <algorithm>
#include <memory>
#include <libimread/IO/gif.hh>
#include <libimread/pixels.hh>

namespace im {
    
    const ImageFormat::options_t GIFFormat::options = GIFFormat::OPTS();
    
    namespace detail {
        
        inline void __attribute__((unused)) setPixel(unsigned char *rgb, int r, int g, int b) {
            rgb[0] = r; rgb[1] = g; rgb[2] = b;
        }
        
        gifholder gifsink(int delay=3) {
            return gifholder(gif::newGIF(delay), gifdisposer<gif::GIF>());
        }
        
        struct gifbuffer {
            int width;
            int height;
            
            public:
                explicit gifbuffer(int w, int h)
                    :data_ptr(std::make_unique<byte[]>(w * h * 3))
                    ,width(w), height(h)
                    {}
                
                virtual ~gifbuffer() { data_ptr.release(); }
                
                byte *data() const { return data_ptr.get(); }
                operator unsigned char *() const { return data(); }
                
            private:
                std::unique_ptr<byte[]> data_ptr;
                gifbuffer(const gifbuffer&);
                gifbuffer(gifbuffer&&);
                gifbuffer &operator=(const gifbuffer&);
                gifbuffer &operator=(gifbuffer&&);
        };
    }
    
    void GIFFormat::write_impl(Image &&input, detail::gifholder &g) {
        
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2); /// should be 3
        const int bit_depth = input.nbits();
        const int full_size = width * height * 3;
        
        /// Allocate buffer
        detail::gifbuffer gbuf(width, height);
        
        /// Check what we got
        if (bit_depth != 8) {
            imread_raise(CannotWriteError,
                    "im::GIFFormat::write() says:   \"UNSUPPORTED IMAGE BIT DEPTH\"",
                 FF("im::GIFFormat::write() got:    `bit_depth` = (int){ %d }", bit_depth),
                    "im::GIFFormat::write() needs:  `bit_depth` = (int){ 8 }");
        }
        if (channels != 3) {
            imread_raise(CannotWriteError,
                    "im::GIFFormat::write() says:   \"UNSUPPORTED IMAGE COLOR MODE\"",
                 FF("im::GIFFormat::write() got:    `channels` = (int){ %d }", channels),
                    "im::GIFFormat::write() needs:  `channels` = (int){ 3 }");
        }
        
        /// Do the pixel loop to interleave RGB data
        byte *data = gbuf.data();
        pix::accessor<byte> at = input.access();
        
        unsigned char *rgb;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                rgb = data + 3 * (width * y + x);
                for (int c = 0; c < channels; c++) {
                    pix::convert(at(x, y, c)[0], rgb[c]);
                }
            }
        }
        
        /// DO IT DOUG
        gif::addFrame(
            g.get(),
            gbuf.width, gbuf.height,
            gbuf.data(), -1); /// delay=-1
    }
    
    void GIFFormat::write(Image &input,
                          byte_sink *output,
                          const options_map &opts) {
        
        /// Do some GIF stuff
        detail::gifholder g = detail::gifsink(3);
        write_impl(std::forward<Image>(input), g);
        
        std::vector<byte> out = gif::write(g.get());
        output->write(&out[0], out.size());
        
        imread_assert(out.size() > 0,
            "gif::write() returned a size-zero byte vector!");
    }
    
    void GIFFormat::write_multi(ImageList &input,
                                byte_sink *output,
                                const options_map &opts) {
        
        /// Do some GIF stuff
        detail::gifholder g = detail::gifsink(3);
        ImageList::vector_t imagevec = input.release();
        std::for_each(imagevec.begin(), imagevec.end(), [&](Image *image) {
            write_impl(std::forward<Image>(*image), g);
        });
        
        std::vector<byte> out = gif::write(g.get());
        output->write(&out[0], out.size());
        
        imread_assert(out.size() > 0,
            "gif::write() returned a size-zero byte vector!");
    }
    
}