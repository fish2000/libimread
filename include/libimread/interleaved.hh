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
#include <libimread/ext/memory/refcount.hh>
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
    using memory::RefCount;
    using memory::DefaultDeleter;
    using memory::ArrayDeleter;
    
    template <typename Color = color::RGBA,
              std::size_t Dimensions = 3>
    class InterleavedImage : public Image, public MetaImage {
        public:
            static constexpr std::size_t C = Color::Meta::channel_count;
            static constexpr std::size_t D = Dimensions;
            using color_t = Color;
            using nonvalue_t = typename Color::NonValue;
            using component_t = typename Color::component_t;
            using channel_t = typename Color::channel_t;
            using composite_t = typename Color::composite_t;
            
            using array_t = std::array<std::size_t, D>;
            using index_t = std::make_index_sequence<D>;
            using bytestring_t = std::basic_string<component_t>;
            
            using channel_list_t = typename Color::channel_list_t;
            using channel_listlist_t = std::initializer_list<channel_list_t>;
            using composite_list_t = std::initializer_list<composite_t>;
            using composite_listlist_t = std::initializer_list<composite_list_t>;
        
        private:
            struct Meta {
                static constexpr std::size_t S = sizeof(component_t);
                std::size_t elem_size;
                array_t extents;
                array_t strides;
                array_t min = { 0, 0, 0 };
                
                explicit Meta(std::size_t x,
                              std::size_t y,
                              std::size_t c = C,
                              std::size_t s = S)
                    :extents(array_t{ x,     y,   c })
                    ,strides(array_t{ x*y*s, x*s, s })
                    ,elem_size(s)
                {}
                
                std::size_t size() const {
                    return size_impl(index_t());
                }
                
                template <std::size_t ...I> inline
                std::size_t size_impl(std::index_sequence<I...>) const {
                    std::size_t out = 1;
                    unpack { (out *= extents[I])... };
                    return out;
                }
                
            };
            
            struct Contents {
                bool dev_dirty;
                bool host_dirty;
                uint64_t dev;
                bytestring_t host;
                
                Contents()
                    :host(), dev(0)
                    ,host_dirty(false), dev_dirty(false)
                    {}
                
                explicit Contents(component_t* bytes, std::size_t size = 0,
                                  uint64_t dev_id = 0)
                    :host(bytes, size), dev(dev_id)
                    ,host_dirty(false), dev_dirty(false)
                    {}
                
                explicit Contents(std::size_t size = 0,
                                  uint64_t dev_id = 0)
                    :host(size, static_cast<component_t>(0)), dev(dev_id)
                    ,host_dirty(false), dev_dirty(false)
                    {}
                
            };
            
            struct HeapContents {
                buffer_t buffer;
                uint8_t *alloc;
                
                explicit HeapContents(buffer_t b, uint8_t *a)
                    :buffer(b), alloc(a)
                    {}
                
                ~HeapContents() { delete[] alloc; }
            };
            
            using RefContents = RefCount<HeapContents>;
            RefContents contents;
            
            void init(int x, int y = 0) {
                buffer_t b = { 0 };
                const int z = C,
                          w = 0;
                b.extent[0] = x;
                b.extent[1] = y;
                b.extent[2] = C;
                b.extent[3] = 0;
                b.stride[0] = x * y * sizeof(composite_t);
                b.stride[1] = x * sizeof(composite_t);
                b.stride[2] = sizeof(composite_t);
                b.stride[3] = 1;
                b.elem_size = sizeof(composite_t);
                
                std::size_t size = x * y;
                uint8_t *ptr = new uint8_t[sizeof(composite_t)*size + 40];
                b.host = ptr;
                b.host_dirty = false;
                b.dev_dirty = false;
                b.dev = 0;
                // while ((std::size_t)b.host & 0x1f) { b.host++; }
                // contents = new Contents(b, b.host);
                contents = RefContents::MakeRef(b, ptr);
            }
            void init(buffer_t b, uint8_t *ptr) {
                contents = RefContents::MakeRef(b, ptr);
            }
            void init(buffer_t b) {
                contents = RefContents::MakeRef(b, b.host);
            }
            
            /// private default constructor
            InterleavedImage(void)
                :Image(), MetaImage(), contents(NULL)
                {}
        
        public:
            
            explicit InterleavedImage(int x, int y)
                :Image(), MetaImage()
                {
                    init(x, y);
                }
            explicit InterleavedImage(buffer_t b)
                :Image(), MetaImage()
                {
                    init(b);
                }
            
            InterleavedImage(const InterleavedImage& other)
                :Image(), MetaImage(), contents(other.contents)
                {}
            
            /// NB: is this really necessary?
            virtual ~InterleavedImage() {
                // contents.release();
            }
            
            InterleavedImage &operator=(const InterleavedImage& other) {
                /// allegedly, the whole 'using-followed-by-naked-swap' crazy talk is a trick:
                /// a ruse to get around the inflexibility of partially-specialized-template bindings
                /// and allow the swap call to get picked up as defined elsewhere -- like for example
                /// the 'friend void RefCount::swap(RefCount&, RefCount&)' func we put in RefCount --
                /// during overload resolution. Which OK yeah if you also think that that
                /// is a fucking weird way to do things then >PPFFFFT< yeah I totally feel you dogg
                using std::swap;
                swap(other.contents, this->contents);
                // contents = other.contents; /// COPY-AND-SWAPDOGG
                return *this;
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
            
            explicit InterleavedImage(channel_t vals[])
                :Image(), MetaImage()
                {
                    init(sizeof(vals) / sizeof(channel_t));
                    for (int idx = 0; idx < sizeof(vals); idx++) {
                        (*this)(idx) = vals[idx];
                    }
                }
            
            explicit InterleavedImage(composite_list_t list)
                :Image(), MetaImage()
                {
                    int idx = 0;
                    init(list.size());
                    for (auto it = list.begin(), item = *it; it != list.end(); ++it) {
                        Color color = Color(static_cast<composite_t>(*it));
                        (*this)(idx) = color.composite;
                        idx++;
                    }
                }
            
            /*
            explicit InterleavedImage(channel_listlist_t list)
                :Image(), MetaImage()
                {
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
            
            /// NB: RETHINK ACCESSORS HERE, MOTHERFUCKER
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
                
                Conversion converter;
                out_t *data = new out_t[size()*size()+40*sizeof(out_t)];
                const int w = width(),
                          h = height();
                
                WTF("Converting...");
                
                /// NB: this next bit is probably way fucked and should get totally rewrote, totally
                out_t *dest;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        typename color_t::array_t components = get(x, y).to_array();
                        dest_color_t dest_color = converter(components.data());
                        dest = data + (y * x);
                        pix::convert(dest_color.composite, *dest);
                    }
                }
                
                WTF("Returning from conversion");
                return (const void*)data;
            };
            
            template <typename DestColor>
            operator InterleavedImage<DestColor>() const {
                using dest_composite_t = typename DestColor::composite_t;
                
                const void* data = conversion_impl<im::color::Convert<Color, DestColor>>();
                
                buffer_t b = {0};
                b.dev = 0;
                b.host = new uint8_t[size()*sizeof(dest_composite_t)+40];
                
                std::memcpy((void *)b.host, (const dest_composite_t*)data, size());
                delete[] (const uint32_t*)data;
                
                b.extent[0] = extent(0);
                b.extent[1] = extent(1);
                b.extent[2] = DestColor::N;
                b.extent[3] = 0;
                b.stride[0] = extent(0) * extent(1) * sizeof(dest_composite_t);
                b.stride[1] = extent(0) * sizeof(dest_composite_t);
                b.stride[2] = sizeof(dest_composite_t);
                b.stride[3] = 1;
                b.host_dirty = true;
                b.dev_dirty = false;
                b.elem_size = sizeof(dest_composite_t);
                
                WTF("Returning from conversion operator");
                return InterleavedImage<DestColor>(b);
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
            
            // int depth = output->dim(2);
            // if (depth == 1) {
            //     InterleavedImage<Color> iimage(
            //         dynamic_cast<InterleavedImage<color::Monochrome>&>(
            //             *output.get()));
            //     iimage.set_host_dirty();
            //     return iimage;
            //
            // } else if (depth == 3) {
            //     InterleavedImage<Color> iimage(
            //         dynamic_cast<InterleavedImage<color::RGB>&>(
            //             *output.get()));
            //     iimage.set_host_dirty();
            //     return iimage;
            // }
            //
            try {
                InterleavedImage<Color> iimage(
                    dynamic_cast<InterleavedImage<color::RGBA>&>(
                        *output.get()));
                iimage.set_host_dirty();
                return iimage;
            } catch (std::bad_cast& exc) {
                WTF("LEAVING ALPHAVILLE.");
                InterleavedImage<Color> iimage(
                    dynamic_cast<InterleavedImage<color::RGB>&>(
                        *output.get()));
                iimage.set_host_dirty();
                return iimage;
            }
            
        }
        
        template <typename Color = color::RGB> inline
        void write(InterleavedImage<Color> *input, const std::string &filename,
                   const options_map &opts = interleaved_default_opts) {
            if (input->dim(2) > 3) { return; }
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write(dynamic_cast<Image&>(input), output.get(), opts);
        }
        
        template <typename Color = color::RGB> inline
        void write(std::unique_ptr<Image> input, const std::string &filename,
                   const options_map &opts = interleaved_default_opts) {
            write<Color>(dynamic_cast<InterleavedImage<Color>>(input.get()),
                         filename, opts);
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