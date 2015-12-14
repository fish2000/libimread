/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_INTERLEAVED_HH_
#define LIBIMREAD_INTERLEAVED_HH_

#include <cstdint>
#include <utility>
#include <limits>
#include <array>
#include <memory>
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
    
    template <std::size_t Dimensions = 3>
    struct Index {
        static constexpr std::size_t D = Dimensions;
        using idx_t = std::ptrdiff_t;
        using idxlist_t = std::initializer_list<idx_t>;
        using array_t = std::array<idx_t, D>;
        using index_t = std::make_index_sequence<D>;
        
        array_t indices;
        
        Index(void)
            :indices{ 0 }
            {}
        
        Index(const idx_t* indexes, std::size_t nd = 0)
            :indices{ 0 }
            {
                if (nd == 0) { nd = D; }
                imread_assert(nd == D,
                              "Dimension mismatch");
                assign_impl(indexes, index_t());
            }
        
        Index(const array_t& indexes)
            :indices{ 0 }
            {
                imread_assert(indexes.size() == D,
                              "Dimension mismatch");
                assign_impl(indexes.data(), index_t());
            }
        
        Index(idxlist_t initlist)
            :indices{ 0 }
            {
                std::size_t idx = 0;
                for (auto it = initlist.begin();
                     it != initlist.end() && idx < D;
                     ++it) { indices[idx] = *it;
                             ++idx; }
            }
        
        Index(const Index& other)
            :Index(other.indices)
            {}
        
        Index& operator=(const Index& other) {
            Index(other).swap(*this);
            return *this;
        }
        
        Index& operator=(idxlist_t initlist) {
            Index(initlist).swap(*this);
            return *this;
        }
        
        // std::size_t ndim() const { return indices.max_size(); }
        constexpr std::size_t ndim() const { return D; }
        idx_t operator[](std::size_t idx) const { return indices[idx]; }
        
        // bool operator==(const Index& other) {
        //     return !std::memcmp(indices.data(),
        //                         other.indices.data(),
        //                         sizeof(idx_t) * D);
        // }
        //
        // bool operator!=(const Index& other) {
        //     return !(*this == other);
        // }
        
        void swap(Index& other) noexcept {
            using std::swap;
            swap(indices, other.indices);
        }
        
        friend void swap(Index& lhs, Index& rhs) noexcept {
            lhs.swap(rhs);
        }
        
        friend std::ostream& operator<<(std::ostream& out, const Index& idx) {
            return out << idx.string_impl(index_t());
        }
        
        bool operator<(const Index& rhs) const { return binary_op<std::less<idx_t>>(rhs.indices); }
        bool operator>(const Index& rhs) const { return binary_op<std::greater<idx_t>>(rhs.indices); }
        bool operator<=(const Index& rhs) const { return binary_op<std::less_equal<idx_t>>(rhs.indices); }
        bool operator>=(const Index& rhs) const { return binary_op<std::greater_equal<idx_t>>(rhs.indices); }
        bool operator==(const Index& rhs) const { return binary_op<std::equal_to<idx_t>>(rhs.indices); }
        bool operator!=(const Index& rhs) const { return binary_op<std::not_equal_to<idx_t>>(rhs.indices); }
        
        private:
            template <typename BinaryPredicate> inline
            bool binary_op(const array_t& rhs,
                           BinaryPredicate predicate = BinaryPredicate()) const {
                return std::equal(std::begin(indices), std::end(indices),
                                  std::begin(rhs),     std::end(rhs),
                                  predicate);
            }
            
            template <std::size_t ...I> inline
            void assign_impl(const idx_t* indexes,
                             std::index_sequence<I...>) const noexcept {
                unpack { (indices[I] = indexes[I])... };
            }
            
            template <std::size_t ...I> inline
            std::string string_impl(std::index_sequence<I...>) const {
                std::string out("( ");
                unpack {
                    (out += std::to_string(indices[I]) +
                            (I == D-1 ? "" : ", "), 0)...
                };
                out += " )";
                return out;
            }
            
    };
    
    template <std::size_t D> inline
    Index<D> operator+=(Index<D>& lhs, const Index<D>& rhs) {
        for (std::size_t idx = 0; idx != D; ++idx) {
            lhs.indices[idx] += rhs.indices[idx];
        }
        return lhs;
    }
    
    template <std::size_t D> inline
    Index<D> operator+(const Index<D>& lhs, const Index<D>& rhs) {
        Index<D> out = lhs;
        out += rhs;
        return out;
    }
    
    template <std::size_t D> inline
    Index<D> operator-=(Index<D>& lhs, const Index<D>& rhs) {
        for (std::size_t idx = 0; idx != D; ++idx) {
            lhs.indices[idx] -= rhs.indices[idx];
        }
        return lhs;
    }
    
    template <std::size_t D> inline
    Index<D> operator-(const Index<D>& lhs, const Index<D>& rhs) {
        Index<D> out = lhs;
        out -= rhs;
        return out;
    }
    
    
    
    struct MetaBase {
        virtual ~MetaBase() {}
    };
    
    template <typename Color = color::RGBA,
              std::size_t Dimensions = 3>
    struct Meta : public MetaBase {
        static constexpr std::size_t S = sizeof(typename Color::channel_t);
        static constexpr std::size_t D = Dimensions;
        friend struct Index<D>;
        using array_t = std::array<std::size_t, D>;
        using index_t = std::make_index_sequence<D>;
        
        std::size_t elem_size;
        array_t extents;
        array_t strides;
        array_t min = { 0 };
        
        Meta(void)
            :extents{ 0 }, strides{ 0 }, elem_size(0)
            {}
        
        explicit Meta(std::size_t x,
                      std::size_t y,
                      std::size_t c = Color::Meta::channel_count,
                      std::size_t s = S)
            :extents{ x,     y,   c }
            ,strides{ x*y*s, x*s, s }
            ,elem_size(s)
            {}
        
        template <typename SourceColor>
        Meta(const Meta<SourceColor>& source)
            :Meta(source.extents[0],
                  source.extents[1],
                  SourceColor::Meta::channel_count,
                  sizeof(typename SourceColor::channel_t))
            {}
        
        std::size_t size() const {
            return size_impl(index_t());
        }
        
        bool is_valid(const Index<D>& idx) const {
            using compare_t = typename Index<D>::idx_t;
            return std::equal(std::begin(idx.indices), std::end(idx.indices),
                              std::begin(extents),     std::end(extents),
                              std::less<compare_t>());
        }
        
        private:
            template <std::size_t ...I> inline
            std::size_t size_impl(std::index_sequence<I...>) const {
                std::size_t out = 1;
                unpack { static_cast<int>(out *= extents[I])... };
                return out;
            }
    };
    
    
    template <typename Color = color::RGBA,
              std::size_t Dimensions = 3>
    class InterleavedImage : public Image, public MetaImage {
        public:
            static constexpr std::size_t C = Color::Meta::channel_count;
            static constexpr std::size_t D = Dimensions;
            static constexpr std::size_t P = 40; /// padding (I still don't quite get this)
            using color_t = Color;
            using nonvalue_t = typename Color::NonValue;
            using component_t = typename Color::component_t; /// == channel_t[N]
            using composite_t = typename Color::composite_t; /// integer-packed components
            using channel_t = typename Color::channel_t; /// single component value
            using array_t = std::array<std::size_t, D>;
            using index_t = std::make_index_sequence<D>;
            
            using bytestring_t = std::basic_string<channel_t>;
            using contents_t = std::shared_ptr<channel_t>;
            using deleter_t = std::default_delete<channel_t[]>;
            using meta_t = Meta<Color, Dimensions>;
            using Contents = contents_t;
            using Meta = meta_t;
            
            using channel_list_t = typename Color::channel_list_t;
            using channel_listlist_t = std::initializer_list<channel_list_t>;
            using composite_list_t = std::initializer_list<composite_t>;
            using composite_listlist_t = std::initializer_list<composite_list_t>;
        
        private:
            
            Contents contents;
            Meta meta;
            
            void init(int x, int y = 1) {
                meta = Meta(x, y);
                contents = contents_t(new channel_t[meta.size()+P],
                                      deleter_t());
            }
            
            void init(Contents c, Meta m) {
                meta = m;
                contents = c;
            }
        
        public:
            
            /// default constructor
            InterleavedImage(void)
                :Image(), MetaImage()
                {}
            
            explicit InterleavedImage(int x, int y)
                :Image(), MetaImage(), meta()
                {
                    init(x, y);
                }
            
            explicit InterleavedImage(Contents c, Meta m)
                :Image(), MetaImage(), contents(c), meta(m)
                {}
            
            InterleavedImage(const InterleavedImage& other)
                :Image(), MetaImage(), contents(other.contents), meta(other.meta)
                {}
            
            /// NB: is this really necessary?
            virtual ~InterleavedImage() {}
            
            InterleavedImage &operator=(const InterleavedImage& other) {
                InterleavedImage(other).swap(*this);
                return *this;
            }
            
            friend void swap(InterleavedImage& lhs, InterleavedImage& rhs) {
                std::exchange(lhs.contents, rhs.contents);
                std::exchange(lhs.meta,     rhs.meta);
            }
            
            void swap(InterleavedImage& other) {
                using std::swap;
                swap(other.contents, this->contents);
                swap(other.meta,     this->meta);
            }
            
            channel_t* data() const {
                return contents.get();
            }
            
            void set_host_dirty(bool dirty = true) {
                imread_raise_default(NotImplementedError);
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
                    init(sizeof(vals) / sizeof(channel_t), 1);
                    for (int idx = 0; idx < sizeof(vals); idx++) {
                        (*this)(idx) = vals[idx];
                    }
                }
            
            explicit InterleavedImage(composite_list_t list)
                :Image(), MetaImage()
                {
                    int idx = 0;
                    init(list.size(), 1);
                    for (auto it = list.begin(), item = *it; it != list.end(); ++it) {
                        Color color = Color(static_cast<composite_t>(*it));
                        (*this)(idx) = color.composite;
                        idx++;
                    }
                }
            
            explicit InterleavedImage(channel_listlist_t list)
                :Image(), MetaImage()
                {
                    int idx = 0;
                    init(list.size(), 1, C);
                    for (auto it = list.begin(), item = *it; it != list.end(); ++it) {
                        for (auto itit = item.begin(); itit != item.end(); ++itit) {
                            (*this)(idx) = static_cast<channel_t>(*itit);
                            ++idx;
                        }
                    }
                }
            
            channel_t &operator()(int x, int y = 0, int z = 0, int w = 0) {
                channel_t *ptr = contents.get();
                x -= meta.min[0];
                y -= meta.min[1];
                z -= meta.min[2];
                // w -= meta.min[3];
                std::size_t s0 = meta.strides[0];
                std::size_t s1 = meta.strides[1];
                std::size_t s2 = meta.strides[2];
                std::size_t s3 = 0; /// LEGACY
                return ptr[s0 * x + s1 * y + s2 * z + s3 * w];
            }
            
            const channel_t &operator()(int x, int y = 0, int z = 0, int w = 0) const {
                channel_t *ptr = contents.get();
                x -= meta.min[0];
                y -= meta.min[1];
                z -= meta.min[2];
                // w -= meta.min[3];
                std::size_t s0 = meta.strides[0];
                std::size_t s1 = meta.strides[1];
                std::size_t s2 = meta.strides[2];
                std::size_t s3 = 0; /// LEGACY
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
                out.composite = static_cast<composite_t>((*this)(x, y));
                return out;
            }
            
            inline const Meta& getMeta() const { return meta; }
            
            /// Halide static image API
            operator buffer_t() const {
                buffer_t b = { 0 };
                b.extent[0] = meta.extents[0];
                b.extent[1] = meta.extents[1];
                b.extent[2] = meta.extents[2];
                b.extent[3] = 0;
                b.stride[0] = meta.strides[0];
                b.stride[1] = meta.strides[1];
                b.stride[2] = meta.strides[2];
                b.stride[3] = 1;
                b.elem_size = meta.elem_size;
                b.dev = 0;
                b.dev_dirty = false;
                b.host_dirty = false;
                b.host = static_cast<uint8_t*>(contents.get());
                return b;
            }
            
            inline const std::size_t size() const { return meta.size(); }
            int dimensions() const { return Dimensions; }
            
            int width() const { return meta.extents[0]; }
            int height() const { return meta.extents[1]; }
            int channels() const { return C; }
            
            int stride_(int dim) const { return meta.strides[dim]; }
            int min(int dim) const { return meta.min[dim]; }
            int extent(int dim) const { return meta.extents[dim]; }
            
            void set_min(int x, int y = 0, int z = 0, int w = 0) {
                meta.min[0] = x;
                meta.min[1] = y;
                meta.min[2] = z;
                meta.min[3] = w;
            }
            
            /// Color conversion
            using toRGB = im::color::Convert<Color,     im::color::RGB>;
            using toRGBA = im::color::Convert<Color,    im::color::RGBA>;
            using toMono = im::color::Convert<Color,    im::color::Monochrome>;
            
            template <typename Conversion,
                      typename Output = typename Conversion::dest_color_t>
            Contents conversion_impl() const {
                using color_t = typename Conversion::color_t;
                using dest_color_t = Output;
                using source_array_t = typename Color::array_t;
                using channel_t = typename Output::channel_t;
                
                Conversion converter;
                Contents destination = contents_t(new channel_t[meta.size()+P],
                                                  deleter_t());
                const int w = width(),
                          h = height();
                
                WTF("Converting...");
                
                channel_t *data = destination.get();
                channel_t *dest;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        source_array_t source_colors = get(x, y).to_array();
                        dest_color_t dest_color = converter(source_colors.data());
                        dest = data + (y * x);
                        pix::convert(dest_color.composite, *dest);
                    }
                }
                
                WTF("Returning from conversion");
                return destination;
            };
            
            template <typename DestColor>
            operator InterleavedImage<DestColor>() const {
                using dest_composite_t = typename DestColor::composite_t;
                using dest_channel_t = typename DestColor::channel_t;
                
                Contents newContents = conversion_impl<im::color::Convert<Color, DestColor>>();
                Meta newMeta(extent(0), extent(1),
                             DestColor::N,
                             sizeof(dest_channel_t));
                
                WTF("Returning from conversion operator");
                return InterleavedImage<DestColor>(newContents, newMeta);
            }
            
            /// im::Image overrides
            virtual const int nbits() const override {
                return sizeof(channel_t) * 8;
            }
            
            virtual const int nbytes() const override {
                return sizeof(channel_t);
            }
            
            virtual int ndims() const override {
                return Dimensions;
            }
            
            virtual int dim(int d) const override {
                return extent(d);
            }
            
            virtual int stride(int s) const override {
                return stride_(s);
            }
            
            inline off_t rowp_stride() const {
                return off_t(meta.strides[1]);
            }
            
            virtual void* rowp(int r) const override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                channel_t* host = data();
                host += off_t(r * meta.strides[1]);
                return static_cast<void*>(host);
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
            void name(std::string& nnm) { nm = nnm; }
            
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
        
        // static options_map interleaved_default_opts;
        
        template <typename Color = color::RGBA>
        InterleavedImage<Color> read(const std::string& filename) {
            const options_map opts;
            InterleavedFactory factory(filename);
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::shared_ptr<Image> output(format->read(input.get(), &factory, opts));
            
            try {
                InterleavedImage<Color> iimage(
                    dynamic_cast<InterleavedImage<color::RGBA>&>(
                        *output.get()));
                return iimage;
            } catch (std::bad_cast& exc) {
                WTF("LEAVING ALPHAVILLE.");
                InterleavedImage<Color> iimage(
                    dynamic_cast<InterleavedImage<color::RGB>&>(
                        *output.get()));
                return iimage;
            }
            
        }
        
        template <typename Color = color::RGBA> inline
        void write(InterleavedImage<Color>* input, const std::string& filename,
                   const options_map& opts) {
            if (input->dim(2) > 3) { return; }
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write(dynamic_cast<Image&>(input), output.get(), opts);
        }
        
        template <typename Color = color::RGBA> inline
        void write(std::unique_ptr<Image> input, const std::string& filename,
                   const options_map& opts) {
            write<Color>(dynamic_cast<InterleavedImage<Color>>(input.get()),
                         filename, opts);
        }
        
        inline void write_multi(ImageList& input, const std::string& filename,
                                const options_map& opts) {
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write_multi(input, output.get(), opts);
        }
        
        template <typename Format, typename Color = color::RGBA> inline
        std::string tmpwrite(InterleavedImage<Color>& input,
                             const options_map& opts) {
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