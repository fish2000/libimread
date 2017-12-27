// Copyright 2015 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#include <algorithm>
#include <memory>
#include <iod/json.hh>
#include <libimread/errors.hh>
#include <libimread/imagelist.hh>
#include <libimread/IO/gif.hh>
#include <libimread/seekable.hh>
#include <libimread/options.hh>

namespace im {
    
    DECLARE_FORMAT_OPTIONS(GIFFormat);
    
    namespace detail {
        
        /// GIF Image pixel-data buffer-babysitter structure --
        /// Wraps a unique_ptr to an allocated byte array with
        /// some sugarary-sweet convenience methods, and a
        /// RAII lifecycle-management guarantee:
        struct gifbuffer {
            
            public:
                explicit gifbuffer(int w, int h, int c = 3)
                    :width(w), height(h), channels(c)
                   { check(w); check(h); check(c); allocate(); }
                
                virtual ~gifbuffer()            {}
                void allocate()                 { data_ptr = 
                                                  std::make_unique<byte[]>(size()); }
                void clear()                    { std::memset(data_ptr.get(), 0, size()); }
                byte* data() const              { return data_ptr.get(); }
                operator unsigned char*() const { return data_ptr.get(); }
                std::size_t size() const        { return width
                                                      * height
                                                    * channels; }
                void check(int dimension) {
                    if (dimension < 1) {
                        imread_raise(GIFIOError,
                                "im::detail::gifbuffer::check() says:   \"UNSUPPORTED DIMENSION VALUE\"",
                             FF("im::detail::gifbuffer::check() got:    `dimension` == (int){ %d }", dimension),
                                "im::detail::gifbuffer::check() needs:  `dimension` >= (int){ 1 }");
                    }
                }
            
            public:
                int width, height, channels;
                gifbuffer(gifbuffer&&) noexcept = default;
            
            protected:
                std::unique_ptr<byte[]> data_ptr;
            
            private:
                gifbuffer(void);
                gifbuffer(gifbuffer const&);
                gifbuffer& operator=(gifbuffer const&);
                gifbuffer& operator=(gifbuffer&&);
            
        };
        
        gifholder gifsink(int frame_interval) {
            return gifholder{ gif::newGIF(frame_interval),
                              gifdisposer<gif::GIF>() };
        }
    
    } /* namespace detail */
    
    void GIFFormat::write_impl(Image const& input, detail::gifholder& g,
                                                   detail::gifbuffer& gbuf,
                                                   int framedelay) { /// framedelay = -1
        
        const int width = input.dim(0);
        const int height = input.dim(1);
        const int channels = input.dim(2); /// should be 3
        const int bit_depth = input.nbits();
        
        /// Check bit depth
        if (bit_depth != 8) {
            imread_raise(GIFIOError,
                    "im::GIFFormat::write() says:   \"UNSUPPORTED IMAGE BIT DEPTH\"",
                 FF("im::GIFFormat::write() got:    `bit_depth` = (int){ %d }", bit_depth),
                    "im::GIFFormat::write() needs:  `bit_depth` = (int){ 8 }");
        }
        
        /// Check channel count
        if (channels != 3) {
            imread_raise(GIFIOError,
                    "im::GIFFormat::write() says:   \"UNSUPPORTED IMAGE COLOR MODE\"",
                 FF("im::GIFFormat::write() got:    `channels` = (int){ %d }", channels),
                    "im::GIFFormat::write() needs:  `channels` = (int){ 3 }");
        }
        
        /// Do the pixel loop to interleave RGB data
        byte* __restrict__ data = gbuf.data();
        unsigned char* __restrict__ rgb;
        
        av::strided_array_view<byte, 3> view = input.view();
        av::strided_array_view<byte, 1> subview;
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                rgb = data + 3 * (width * y + x);
                subview = view[x][y];
                for (int c = 0; c < channels; ++c) {
                    rgb[c] = subview[c];
                }
            }
        }
        
        /// DO IT DOUG
        gif::addFrame(g.get(), gbuf.width,
                               gbuf.height,
                               gbuf.data(), framedelay);
    }
    
    void GIFFormat::write(Image& input,
                          byte_sink* output,
                          Options const& opts) {
        
        /// Do some GIF stuff
        detail::gifholder g = detail::gifsink();
        detail::gifbuffer b = detail::gifbuffer(input.width(),
                                                input.height(),
                                                input.planes());
        write_impl(input, g, b);
        
        bytevec_t out = gif::write(g.get());
        output->write(out);
        output->flush();
        
        // imread_assert(out.size() > 0,
        //     "gif::write() returned a size-zero byte vector!");
    }
    
    void GIFFormat::write_multi(ImageList& input,
                                byte_sink* output,
                                Options const& opts) {
        /// Pre-compute ImageList sizes
        input.compute_sizes();
        
        /// Sort out frame delay
        int delay = opts.cast<int>("gif:delay", GIFFormat::options.delay);
        
        /// Do some GIF stuff
        detail::gifholder g = detail::gifsink(delay);
        detail::gifbuffer b = detail::gifbuffer(input.width(),
                                                input.height(),
                                                input.planes());
        
        std::for_each(input.begin(), input.end(),
                  [&](Image* image) { write_impl(*image, g, b, delay); });
        
        bytevec_t out = gif::write(g.get());
        output->write(out);
        output->flush();
        
        // imread_assert(out.size() > 0,
        //     "gif::write() returned a size-zero byte vector!");
    }
    
}