// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#include <libimread/IO/gif.hh>

namespace im {
    
    namespace {
        
        inline void __attribute__((unused)) setPixel(unsigned char *rgb, int r, int g, int b) {
            rgb[0] = r; rgb[1] = g; rgb[2] = b;
        }
        
        template <typename T = byte>
        inline T *at(Image &im, int x, int y, int z) {
            return &im.rowp_as<T>(0)[x*im.stride(0) +
                                     y*im.stride(1) +
                                     z*im.stride(2)];
        }
        
        struct gif_buffer {
            public:
                gif_buffer(int w, int h)
                    :data_ptr(new byte[width * height * 3])
                    ,width(w), height(h)
                    {}
            
                virtual ~gif_buffer() {
                    delete[] data_ptr;
                }
            
                byte *data() const { return data_ptr; }
                operator unsigned char *() const { return data(); }
            
                void add_to(gif::GIF *g, int delay=-1) {
                    gif::addFrame(g, width, height, data(), delay);
                }
            
                const int width;
                const int height;
            
            private:
                byte *data_ptr;
        };
        
        struct gif_holder {
            public:
                gif_holder(int delay=3)
                    :gif_struct(gif::newGIF(delay))
                    {}
                
                virtual ~gif_holder() {
                    gif::dispose(gif_struct);
                    gif_struct = NULL;
                }
                
                void add_frame(gif_buffer gbuf, int delay=-1) {
                    gif::addFrame(gif_struct, gbuf.width, gbuf.height, gbuf.data(), delay);
                }
                
                std::vector<byte> write() { return gif::write(gif_struct); }
                
            private:
                gif::GIF *gif_struct;
        };
    
    }
    
    void GIFFormat::write(Image &input,
            byte_sink *output,
            const options_map &opts) {
        
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2); /// should be 3
        const int bit_depth = input.nbits();
        const int full_size = width * height * 3;
        
        /// Do some GIF stuff
        gif_holder g;
        gif_buffer gbuf(width, height);
        
        /// Check what we got
        if (bit_depth != 8) {
            throw CannotWriteError(
                "ERROR:",
                    "im::GIFFormat::write() says:   \"UNSUPPORTED IMAGE BIT DEPTH\"",
                 FF("im::GIFFormat::write() got:    `bit_depth` = (int){ %d }", bit_depth),
                    "im::GIFFormat::write() needs:  `bit_depth` = (int){ 8 }"
            );
        }
        if (channels != 3) {
            throw CannotWriteError(
                "ERROR:",
                    "im::GIFFormat::write() says:   \"UNSUPPORTED IMAGE COLOR MODE\"",
                 FF("im::GIFFormat::write() got:    `channels` = (int){ %d }", channels),
                    "im::GIFFormat::write() needs:  `channels` = (int){ 3 }"
            );
        }
        
        /// Do the pixel loop to interleave RGB data
        byte *data = gbuf.data();
        unsigned char *rgb;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                rgb = data + 3 * (width * y + x);
                for (int c = 0; c < channels; c++) {
                    pix::convert(at(input, x, y, c)[0], rgb[c]);
                }
            }
        }
        
        /// DO IT DOUG
        g.add_frame(gbuf);
        (*output) << g.write();
    }
    
}