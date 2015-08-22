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
                buffer.extent[2] = C;
                buffer.extent[3] = 0;
                buffer.stride[0] = x * y * sizeof(composite_t);
                buffer.stride[1] = x * sizeof(composite_t);
                buffer.stride[2] = sizeof(composite_t);
                buffer.stride[3] = 1;
                buffer.elem_size = sizeof(composite_t);
                
                std::size_t size = x * y;
                uint8_t *ptr = new uint8_t[sizeof(composite_t)*size + 40];
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
            
            composite_t *data() const {
                return reinterpret_cast<composite_t*>(contents->buffer.host);
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
                init(list.size());
                for (auto it = list.begin(), item = *it; it != list.end(); ++it) {
                    Color color = Color(static_cast<composite_t>(*it));
                    (*this)(idx) = color.composite;
                    idx++;
                }
            }
            
            /*
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
            */
            
            composite_t &operator()(int x, int y = 0, int z = 0, int w = 0) {
                composite_t *ptr = reinterpret_cast<composite_t*>(contents->buffer.host);
                x -= contents->buffer.min[0];
                y -= contents->buffer.min[1];
                // z -= contents->buffer.min[2];
                // w -= contents->buffer.min[3];
                std::size_t s0 = contents->buffer.stride[0];
                std::size_t s1 = contents->buffer.stride[1];
                //std::size_t s2 = contents->buffer.stride[2];
                //std::size_t s3 = contents->buffer.stride[3];
                std::size_t s2 = 0;
                std::size_t s3 = 0;
                return ptr[s0 * x + s1 * y + s2 * z + s3 * w];
            }
            
            const composite_t &operator()(int x, int y = 0, int z = 0, int w = 0) const {
                const composite_t *ptr = reinterpret_cast<const composite_t*>(contents->buffer.host);
                x -= contents->buffer.min[0];
                y -= contents->buffer.min[1];
                z -= contents->buffer.min[2];
                w -= contents->buffer.min[3];
                std::size_t s0 = contents->buffer.stride[0];
                std::size_t s1 = contents->buffer.stride[1];
                // std::size_t s2 = contents->buffer.stride[2];
                // std::size_t s3 = contents->buffer.stride[3];
                std::size_t s2 = 0;
                std::size_t s3 = 0;
                return ptr[s0 * x + s1 * y + s2 * z + s3 * w];
            }
            
            void set(int x, int y, const Color& color) {
                (*this)(x, y) = color.composite;
            }
            void set(int x, int y, composite_t composite) {
                (*this)(x, y) = composite;
            }
            void set(int x, int y, channel_list_t&& list) {
                set(x, y, Color(std::forward<channel_list_t>(list)));
            }
            void set(int x, int y, array_t&& array) {
                set(x, y, Color(std::forward<array_t>(array)));
            }
            
            inline Color get(int x, int y) const {
                Color out;
                out.composite = (*this)(x, y);
                return out;
            }
            
            const std::size_t size() const{
                return contents->buffer.extent[0] *
                       contents->buffer.extent[1] *
                       contents->buffer.extent[2] *
                       sizeof(composite_t);
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
                return C;
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
            
            template <typename T = byte> inline
            pix::accessor<T> access() const {
                return pix::accessor<T>(
                    static_cast<T*>(rowpc(0)), stride(0),
                                               stride(1),
                                               stride(2));
            }
            
            template <typename Conversion,
                      typename Output = typename Conversion::dest_color_t::composite_t>
            const void* conversion_impl() const {
                using color_t = typename Conversion::color_t;
                using dest_color_t = typename Conversion::dest_color_t;
                using source_component_t = typename color_t::component_t;
                using in_t = typename color_t::composite_t;
                using out_t = Output;
                
                WTF("convert() called");
                
                Conversion converter;
                out_t *data = new out_t[size()*size()+40*sizeof(out_t)];
                
                const int w = width(),
                          h = height();
                
                out_t *dest;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        typename color_t::array_t components = get(x, y).to_array();
                        WTF("in pixel loop", FF("x = %i, y = %i, components.size() = %i",
                            x, y, components.size()
                        ));
                        dest_color_t dest_color = converter(components.data());
                        WTF("Returned from converter to pixel loop");
                        dest = data + (y * x);
                        WTF("About to call pix::convert()");
                        pix::convert(dest_color.composite, *dest);
                    }
                }
                
                WTF("convert() returning");
                return (const void*)data;
            };
            
            template <typename DestColor, typename = void>
            operator InterleavedImage<DestColor>() const {
                using dest_composite_t = typename DestColor::composite_t;
                const void* data = conversion_impl<im::color::Convert<Color, DestColor>>();
                buffer_t buffer = {0};
                buffer.dev = 0;
                //buffer.host = reinterpret_cast<uint8_t*>(data);
                std::memcpy(buffer.host, data, size());
                delete[] (const uint32_t*)data;
                buffer.extent[0] = extent(0);
                buffer.extent[1] = extent(1);
                buffer.extent[2] = DestColor::N;
                buffer.extent[3] = 0;
                buffer.stride[0] = extent(0) * extent(1) * sizeof(dest_composite_t);
                buffer.stride[1] = extent(0) * sizeof(dest_composite_t);
                buffer.stride[2] = sizeof(dest_composite_t);
                buffer.stride[3] = 1;
                buffer.host_dirty = true;
                buffer.dev_dirty = false;
                buffer.elem_size = sizeof(dest_composite_t);
                return InterleavedImage<DestColor>(buffer);
            }
            
            /// im::Image overrides
            virtual const int nbits() const override {
                return sizeof(channel_t) * 8;
            }
            
            virtual const int nbytes() const override {
                return sizeof(channel_t);
            }
            
            virtual int ndims() const override {
                //return InterleavedImage<Color>::dimensions();
                return 3;
            }
            
            virtual int dim(int d) const override {
                return InterleavedImage<Color>::extent(d);
            }
            
            virtual int stride(int s) const override {
                return InterleavedImage<Color>::stride_(s);
            }
            
            inline off_t rowp_stride() const {
                return off_t(InterleavedImage<Color>::stride(1));
            }
            
            virtual void *rowp(int r) override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                channel_t *host = reinterpret_cast<channel_t*>(InterleavedImage<Color>::data());
                host += off_t(r * rowp_stride());
                return static_cast<void *>(host);
            }
            
            void *rowpc(int r) const {
                channel_t *host = reinterpret_cast<channel_t*>(InterleavedImage<Color>::data());
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
        
        template <typename Color = color::RGBA>
        InterleavedImage<Color> read(const std::string &filename,
                                     const options_map &opts = interleaved_default_opts) {
            InterleavedFactory factory(filename);
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            
            // WTF("About to apply dynamic_cast<HybridImage<T>&> to output:",
            //     FF("WIDTH = %i, HEIGHT = %i, DEPTH = %i, NDIMS = %i", output->dim(0),
            //                                                           output->dim(1),
            //                                                           output->dim(2),
            //                                                           output->ndims()));
            
            int depth = output->dim(2);
            
            if (depth == 1) {
                InterleavedImage<Color> iimage(
                    dynamic_cast<InterleavedImage<color::Monochrome>&>(
                        *output.get()));
                iimage.set_host_dirty();
                return iimage;
            
            } else if (depth == 3) {
                InterleavedImage<Color> iimage(
                    dynamic_cast<InterleavedImage<color::RGB>&>(
                        *output.get()));
                iimage.set_host_dirty();
                return iimage;
            }
            
            InterleavedImage<Color> iimage(
                dynamic_cast<InterleavedImage<color::RGBA>&>(
                    *output.get()));
            iimage.set_host_dirty();
            return iimage;
            
        }
        
        template <typename Color = color::RGB> inline
        void write(InterleavedImage<Color> &input, const std::string &filename,
                   const options_map &opts = interleaved_default_opts) {
            if (input.dim(2) > 3) { return; }
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write(dynamic_cast<Image&>(input), output.get(), opts);
        }
        
        inline void write_multi(ImageList &input, const std::string &filename,
                                const options_map &opts = interleaved_default_opts) {
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write_multi(input, output.get(), opts);
        }
        
        template <typename Format, typename Color = color::RGB> inline
        std::string tmpwrite(InterleavedImage<Color> &input,
                             const options_map &opts = interleaved_default_opts) {
            if (input.dim(2) > 3) { return ""; }
            im::fs::NamedTemporaryFile tf(Format::get_suffix());
            std::unique_ptr<ImageFormat> format(new Format);
            std::unique_ptr<FileSink> output(new FileSink(tf.str()));
            format->write(dynamic_cast<Image&>(input), output.get(), opts);
            return tf.str();
        }
        
    }
    
    
}

#endif /// LIBIMREAD_INTERLEAVED_HH_