/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INTERLEAVED_HH_
#define LIBIMREAD_INTERLEAVED_HH_

#include <cstdint>
#include <utility>
#include <limits>
#include <array>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <type_traits>
#include <initializer_list>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/private/buffer_t.h>
#include <libimread/color.hh>
#include <libimread/pixels.hh>
#include <libimread/fs.hh>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>
#include <libimread/formats.hh>
#include <libimread/options.hh>

namespace im {
    
    using MetaImage = ImageWithMetadata;
    
    template <typename Color = color::RGBA>
    class InterleavedImage : public Image, public MetaImage {
        public:
            static constexpr std::size_t C = Color::Meta::channel_count;
            using color_t = Color;
            using nonvalue_t = typename Color::NonValue;
            using component_t = typename Color::component_t;
            using channel_t = typename Color::channel_t;
            using composite_t = typename Color::composite_t;
            
            using array_t = std::array<channel_t, C>;
            using index_t = std::make_index_sequence<C>;
            
            using channel_list_t = typename Color::channel_list_t;
            using channel_listlist_t = std::initializer_list<channel_list_t>;
            using composite_list_t = std::initializer_list<composite_t>;
            using composite_listlist_t = std::initializer_list<composite_list_t>;
        
        private:
            struct Contents {
                buffer_t buffer;
                int32_t refcount;
                uint8_t *alloc;
                
                Contents(buffer_t b, uint8_t *a)
                    :buffer(b), refcount(1), alloc(a)
                    {}
                
                ~Contents() { delete[] alloc; }
            };
            
            Contents *contents;
            
            void init(int x, int y = 0) {
                buffer_t buffer = { 0 };
                const int z = C,
                          w = 0;
                buffer.extent[0] = x;
                buffer.extent[1] = y;
                buffer.extent[2] = z;
                buffer.extent[3] = w;
                buffer.stride[0] = x * y * sizeof(channel_t);
                buffer.stride[1] = x * sizeof(channel_t);
                buffer.stride[2] = sizeof(channel_t);
                buffer.stride[3] = 1;
                buffer.elem_size = sizeof(channel_t);
                
                std::size_t size = x * y * z;
                uint8_t *ptr = new uint8_t[sizeof(channel_t)*size + 40];
                buffer.host = ptr;
                buffer.host_dirty = false;
                buffer.dev_dirty = false;
                buffer.dev = 0;
                while ((std::size_t)buffer.host & 0x1f) { buffer.host++; }
                contents = new Contents(buffer, buffer.host);
            }
            
            void init(buffer_t buffer) {
                contents = new Contents(buffer, buffer.host);
            }
        
        public:
            InterleavedImage()
                :contents(NULL)
                {}
            
            explicit InterleavedImage(int x, int y)      { init(x, y); }
            explicit InterleavedImage(buffer_t buffer)   { init(buffer); }
            
            InterleavedImage(const InterleavedImage& other)
                :contents(other.contents)
                {
                    incref(contents);
                }
            InterleavedImage(InterleavedImage&& other)
                :contents(other.contents)
                {
                    ref(contents, 1);
                    other.contents = NULL;
                }
            
            virtual ~InterleavedImage() { decref(contents); }
            
            InterleavedImage &operator=(const InterleavedImage& other) {
                Contents *p = other.contents;
                incref(p);
                decref(contents);
                contents = p;
                return *this;
            }
            
            InterleavedImage &operator=(InterleavedImage&& other) {
                Contents *p = other.contents;
                ref(p, 1);
                decref(contents);
                contents = p;
                other.contents = NULL;
                return *this;
            }
            
            inline void ref(Contents *c, int count = 1) {
                if (c) { c->refcount = count; }
            }
            
            inline void incref(Contents *c) {
                if (c) { c->refcount++; }
            }
            
            inline void decref(Contents *c) {
                if (c) {
                    c->refcount--;
                    if (c->refcount == 0) {
                        delete c;
                        c = NULL;
                    }
                }
            }
            
            channel_t *data() const {
                return static_cast<channel_t*>(contents->buffer.host);
            }
            
            void set_host_dirty(bool dirty = true) {
                contents->buffer.host_dirty = dirty;
            }
            
            void copy_to_host() {
                imread_raise_default(NotImplementedError);
            }
            void copy_to_dev() {
                imread_raise_default(NotImplementedError);
            }
            void dev_free() {
                imread_raise_default(NotImplementedError);
            }
            
            explicit InterleavedImage(channel_t vals[]) {
                init(sizeof(vals) / sizeof(channel_t));
                for (int idx = 0; idx < sizeof(vals); idx++) {
                    (*this)(idx) = vals[idx];
                }
            }
            
            explicit InterleavedImage(composite_list_t list) {
                int idx = 0;
                init(list.size() * C);
                for (auto it = list.begin(), item = *it; it != list.end(); ++it) {
                    Color color = Color(static_cast<composite_t>(*it));
                    for (int i = 0; i < C; ++i) {
                        (*this)(idx+i) = color.components[i];
                    }
                    idx += C;
                }
            }
            
            explicit InterleavedImage(channel_listlist_t list) {
                int idx = 0;
                init(list.size() * C);
                for (auto it = list.begin(), item = *it; it != list.end(); ++it) {
                    for (auto itit = item.begin(); itit != item.end(); ++itit) {
                        (*this)(idx) = static_cast<channel_t>(*itit);
                        ++idx;
                    }
                }
            }
            
            channel_t &operator()(int x, int y = 0, int z = 0, int w = 0) {
                channel_t *ptr = static_cast<channel_t*>(contents->buffer.host);
                x -= contents->buffer.min[0];
                y -= contents->buffer.min[1];
                z -= contents->buffer.min[2];
                w -= contents->buffer.min[3];
                std::size_t s0 = contents->buffer.stride[0];
                std::size_t s1 = contents->buffer.stride[1];
                std::size_t s2 = contents->buffer.stride[2];
                std::size_t s3 = contents->buffer.stride[3];
                return ptr[s0 * x + s1 * y + s2 * z + s3 * w];
            }
            
            const channel_t &operator()(int x, int y = 0, int z = 0, int w = 0) const {
                const channel_t *ptr = static_cast<const channel_t*>(contents->buffer.host);
                x -= contents->buffer.min[0];
                y -= contents->buffer.min[1];
                z -= contents->buffer.min[2];
                w -= contents->buffer.min[3];
                std::size_t s0 = contents->buffer.stride[0];
                std::size_t s1 = contents->buffer.stride[1];
                std::size_t s2 = contents->buffer.stride[2];
                std::size_t s3 = contents->buffer.stride[3];
                return ptr[s0 * x + s1 * y + s2 * z + s3 * w];
            }
            
            void set(int x, int y, const Color& color) {
                for (int i = 0; i < C; ++i) {
                    (*this)(x, y, i) = color.components[i];
                }
            }
            void set(int x, int y, composite_t composite) {
                set(x, y, Color(composite));
            }
            void set(int x, int y, channel_list_t&& list) {
                set(x, y, Color(std::forward<channel_list_t>(list)));
            }
            void set(int x, int y, array_t&& array) {
                set(x, y, Color(std::forward<array_t>(array)));
            }
            
            Color get(int x, int y) {
                Color out;
                for (int i = 0; i < C; ++i) {
                    out.components[i] = *this->operator()(x, y, i);
                }
                return out;
            }
            
            const Color& get(int x, int y) const {
                Color out;
                for (int i = 0; i < C; ++i) {
                    out.components[i] = *this->operator()(x, y, i);
                }
                return out;
            }
            
            const std::size_t size() const{
                return contents->buffer.extent[0] *
                       contents->buffer.extent[1] *
                       contents->buffer.extent[2];
            }
            
            /// Halide static image API
            operator buffer_t *() const {
                return &(contents->buffer);
            }
            
            int dimensions() const { return 3; }
            
            int width() const {
                return contents->buffer.extent[0];
            }
            
            int height() const {
                return contents->buffer.extent[1];
            }
            
            int channels() const {
                return contents->buffer.extent[2];
            }
            
            int stride_(int dim) const {
                return contents->buffer.stride[dim];
            }
            
            int min(int dim) const {
                return contents->buffer.min[dim];
            }
            
            int extent(int dim) const {
                return contents->buffer.extent[dim];
            }
            
            void set_min(int x, int y = 0, int z = 0, int w = 0) {
                contents->buffer.min[0] = x;
                contents->buffer.min[1] = y;
                contents->buffer.min[2] = z;
                contents->buffer.min[3] = w;
            }
            
            /// Color conversion
            using toRGB = im::color::Convert<Color,     im::color::RGB>;
            using toRGBA = im::color::Convert<Color,    im::color::RGBA>;
            using toMono = im::color::Convert<Color,    im::color::Monochrome>;
            
            template <typename Conversion,
                      typename Output = typename Conversion::dest_color_t::channel_t>
            std::shared_ptr<Output> convert() {
                using color_t = typename Conversion::color_t;
                using dest_color_t = typename Conversion::dest_color_t;
                using source_component_t = typename color_t::component_t;
                using in_t = typename color_t::channel_t;
                using out_t = Output;
                
                Conversion converter;
                std::shared_ptr<out_t> out(new out_t[sizeof(out_t)*size()+40]);
                out_t *data = out.get();
                
                const int w = width(),
                          h = height(),
                          c = dest_color_t::channels(),
                          siz = sizeof(out_t) * c;
                
                pix::accessor<in_t> at = access();
                pix::accessor<out_t> to = pix::accessor<out_t>(data,
                                                               w * h, w, 1);
                
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        dest_color_t dest_color = converter((source_component_t)at(x, y, 0));
                        std::memcpy(to(x, y, 0),
                                    &dest_color.components[0],
                                    siz);
                    }
                }
                
                return out;
            };
            
            template <typename DestColor>
            operator InterleavedImage<DestColor>() const {
                auto data = convert<im::color::Convert<Color, DestColor>>();
                buffer_t buffer = {0};
                buffer.dev = 0;
                buffer.host = data.release();
                buffer.extent[0] = extent(0);
                buffer.extent[1] = extent(1);
                buffer.extent[2] = DestColor::N;
                buffer.extent[3] = extent(3);
                buffer.stride[0] = extent(0) * extent(1) * sizeof(DestColor::channel_t);
                buffer.stride[1] = extent(0) * sizeof(DestColor::channel_t);
                buffer.stride[2] = sizeof(DestColor::channel_t);
                buffer.stride[3] = 1;
                buffer.host_dirty = true;
                buffer.dev_dirty = false;
                buffer.elem_size = sizeof(DestColor::channel_t);
                return InterleavedImage<DestColor>(buffer);
            }
            
            template <>
            operator InterleavedImage<Color>() const {
                return *this;
            }
            
            /// im::Image overrides
            virtual const int nbits() const override {
                return sizeof(channel_t) * 8;
            }
            
            virtual const int nbytes() const override {
                return sizeof(channel_t);
            }
            
            virtual int ndims() const override {
                return InterleavedImage<Color>::dimensions();
            }
            
            virtual int dim(int d) const override {
                return InterleavedImage<Color>::extent(d);
            }
            
            virtual int stride(int s) const override {
                return InterleavedImage<Color>::stride_(s);
            }
            
            inline off_t rowp_stride() const {
                return InterleavedImage<Color>::channels() == 1 ? 0 : off_t(InterleavedImage<Color>::stride(0));
            }
            
            virtual void *rowp(int r) override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                channel_t *host = InterleavedImage<Color>::data();
                host += off_t(r * rowp_stride());
                return static_cast<void *>(host);
            }
            
    };
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    class InterleavedFactory : public ImageFactory {
        
        private:
            std::string nm;
        
        public:
            InterleavedFactory()
                :nm(std::string(""))
                {}
            InterleavedFactory(const std::string &n)
                :nm(std::string(n))
                {}
            
            virtual ~InterleavedFactory() {}
            
            std::string &name() { return nm; }
            void name(std::string &nnm) { nm = nnm; }
            
        protected:
            virtual std::unique_ptr<Image> create(int nbits,
                                                  int xHEIGHT, int xWIDTH,
                                                  int xDEPTH,
                                                  int d3 = 0, int d4 = 0) override {
                if (xDEPTH == 1) {
                    return std::unique_ptr<Image>(
                        new InterleavedImage<color::Monochrome>(xWIDTH, xHEIGHT));
                } else if (xDEPTH == 3) {
                    return std::unique_ptr<Image>(
                        new InterleavedImage<color::RGB>(xWIDTH, xHEIGHT));
                } else if (xDEPTH == 4) {
                    return std::unique_ptr<Image>(
                        new InterleavedImage<color::RGBA>(xWIDTH, xHEIGHT));
                } else {
                    return std::unique_ptr<Image>(
                        new InterleavedImage<color::RGBA>(xWIDTH, xHEIGHT));
                }
            }
            
            virtual std::shared_ptr<Image> shared(int nbits,
                                                  int xHEIGHT, int xWIDTH,
                                                  int xDEPTH,
                                                  int d3 = 0, int d4 = 0) override {
                if (xDEPTH == 1) {
                    return std::shared_ptr<Image>(
                        new InterleavedImage<color::Monochrome>(xWIDTH, xHEIGHT));
                } else if (xDEPTH == 3) {
                    return std::shared_ptr<Image>(
                        new InterleavedImage<color::RGB>(xWIDTH, xHEIGHT));
                } else if (xDEPTH == 4) {
                    return std::shared_ptr<Image>(
                        new InterleavedImage<color::RGBA>(xWIDTH, xHEIGHT));
                } else {
                    return std::shared_ptr<Image>(
                        new InterleavedImage<color::RGBA>(xWIDTH, xHEIGHT));
                }
            }
    };

#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
    
    
    namespace interleaved {
        
        static const options_map interleaved_default_opts;
        
        template <typename Color = im::color::RGBA>
        im::InterleavedImage<Color> read(const std::string &filename,
                                         const options_map &opts = interleaved_default_opts) {
            im::InterleavedFactory factory(filename);
            std::unique_ptr<im::ImageFormat> format(im::for_filename(filename));
            std::unique_ptr<im::FileSource> input(new im::FileSource(filename));
            std::unique_ptr<im::Image> output = format->read(input.get(), &factory, opts);
            im::InterleavedImage<im::color::RGBA> rgbaimage(
                dynamic_cast<im::InterleavedImage<im::color::RGBA>&>(
                    *output.release()));
            rgbaimage.set_host_dirty();
            return rgbaimage;
        }
        
        template <typename Color = im::color::RGB> inline
        void write(im::InterleavedImage<Color> &input, const std::string &filename,
                                                   const options_map &opts = interleaved_default_opts) {
            if (input.dim(2) > 3) { return; }
            std::unique_ptr<im::ImageFormat> format(im::for_filename(filename));
            std::unique_ptr<im::FileSink> output(new im::FileSink(filename));
            format->write(dynamic_cast<im::Image&>(input), output.get(), opts);
        }
        
        inline void write_multi(im::ImageList &input, const std::string &filename,
                                           const options_map &opts = interleaved_default_opts) {
            std::unique_ptr<im::ImageFormat> format(im::for_filename(filename));
            std::unique_ptr<im::FileSink> output(new im::FileSink(filename));
            format->write_multi(input, output.get(), opts);
        }
        
        template <typename Format, typename Color = im::color::RGB> inline
        std::string tmpwrite(im::InterleavedImage<Color> &input,
                             const options_map &opts = interleaved_default_opts) {
            if (input.dim(2) > 3) { return ""; }
            im::fs::NamedTemporaryFile tf(Format::get_suffix());
            std::unique_ptr<im::ImageFormat> format(new Format);
            std::unique_ptr<im::FileSink> output(new im::FileSink(tf.str()));
            format->write(dynamic_cast<im::Image&>(input), output.get(), opts);
            return tf.str();
        }
        
    }
    
    
}

#endif /// LIBIMREAD_INTERLEAVED_HH_