/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HALIDE_HH_
#define LIBIMREAD_HALIDE_HH_

#include <iostream>
#include <memory>
#include <utility>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <type_traits>

#include <Halide.h>
#define BUFFER_T_DEFINED

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/errors.hh>
#include <libimread/file.hh>
#include <libimread/filehandle.hh>
#include <libimread/image.hh>
#include <libimread/metadata.hh>
#include <libimread/imagelist.hh>
#include <libimread/formats.hh>
#include <libimread/options.hh>

namespace im {
    
    namespace detail {
        
        template <typename PixelType>
        struct for_type;
        
        #define DEFINE_TYPE_MAPPING(PixelType, StructType)                              \
        template <>                                                                     \
        struct for_type<PixelType> {                                                    \
            using type = PixelType;                                                     \
            static Halide::Type get() { return Halide::StructType; }                    \
        };
        
        DEFINE_TYPE_MAPPING(bool,           Bool());
        DEFINE_TYPE_MAPPING(uint8_t,        UInt(8));
        DEFINE_TYPE_MAPPING(uint16_t,       UInt(16));
        DEFINE_TYPE_MAPPING(uint32_t,       UInt(32));
        DEFINE_TYPE_MAPPING(uint64_t,       UInt(64));
        DEFINE_TYPE_MAPPING(int8_t,         Int(8));
        DEFINE_TYPE_MAPPING(int16_t,        Int(16));
        DEFINE_TYPE_MAPPING(int32_t,        Int(32));
        DEFINE_TYPE_MAPPING(int64_t,        Int(64));
        DEFINE_TYPE_MAPPING(float,          Float(32));
        DEFINE_TYPE_MAPPING(double,         Float(64));
        DEFINE_TYPE_MAPPING(long double,    Float(64));
        
        template <typename PointerBase>
        struct for_type<PointerBase*> {
            using type = std::add_pointer_t<PointerBase>;
            static Halide::Type get() { return Halide::Handle(); }
        };
        
        template <typename PixelType> inline
        Halide::Type halide_t() { return for_type<std::remove_cv_t<
                                                  std::decay_t<PixelType>>>::get(); }
        
    } /* namespace detail */
    
#define CALCULATE_ROWP_STRIDE() (halide_image_t::channels() == 1 ? 0 : static_cast<off_t>(halide_image_t::stride(1)))
    
    template <typename T>
    using HalImage = Halide::Buffer<std::decay_t<T>>;
    
    template <typename pT,
              typename hT = HalImage<pT>>
    class HybridImage final : public hT, public Image {
        
        public:
            using pixel_t = pT;
            using halide_image_t = hT;
            
        public:
            HybridImage()
                :halide_image_t(), Image()
                { rowp_stride = CALCULATE_ROWP_STRIDE(); }
            
            HybridImage(int x, int y, int z, int w, std::string const& name="")
                :halide_image_t(x, y, z, w), Image()
                { rowp_stride = CALCULATE_ROWP_STRIDE(); }
            
            HybridImage(int x, int y, int z, std::string const& name="")
                :halide_image_t(x, y, z), Image()
                { rowp_stride = CALCULATE_ROWP_STRIDE(); }
            
            HybridImage(int x, int y, std::string const& name="")
                :halide_image_t(x, y), Image()
                { rowp_stride = CALCULATE_ROWP_STRIDE(); }
            
            HybridImage(int x, std::string const& name="")
                :halide_image_t(x), Image()
                { rowp_stride = CALCULATE_ROWP_STRIDE(); }
            
            HybridImage(buffer_t const* b, std::string const& name="")
                :halide_image_t(b, name), Image()
                { rowp_stride = CALCULATE_ROWP_STRIDE(); }
            
            /// TODO: halide_buffer_t !
            
        public:
            /// copy constructor
            HybridImage(HybridImage const& other)
                :halide_image_t(dynamic_cast<halide_image_t const&>(other))
                ,Image(dynamic_cast<Image const&>(other))
                { rowp_stride = CALCULATE_ROWP_STRIDE(); }
            
        public:
            /// move constructor
            HybridImage(HybridImage&& other) noexcept
                :halide_image_t(dynamic_cast<halide_image_t&&>(other))
                ,Image(dynamic_cast<Image&&>(other))
                { rowp_stride = CALCULATE_ROWP_STRIDE(); }
            
        public:
            using halide_image_t::operator();
            using halide_image_t::dimensions;
            using halide_image_t::extent;
            using halide_image_t::stride;
            using halide_image_t::channels;
            using halide_image_t::data;
            using halide_image_t::type;
            
        public:
            virtual ~HybridImage() {}
            
        public:
            Halide::Type type() const {
                return halide_image_t::type();
            }
            
        public:
            virtual int nbits() const override {
                return sizeof(pT) * 8;
            }
            
            virtual int nbytes() const override {
                return sizeof(pT);
            }
            
            virtual int ndims() const override {
                return halide_image_t::dimensions();
            }
            
            virtual int dim(int d) const override {
                return halide_image_t::extent(d);
            }
            
            virtual int stride(int s) const override {
                return halide_image_t::stride(s);
            }
            
            virtual int min(int s) const override {
                return halide_image_t::min(s);
            }
            
            virtual bool is_signed() const override {
                return std::is_signed<pT>::value;
            }
            
            virtual bool is_floating_point() const override {
                return std::is_floating_point<pT>::value;
            }
            
            virtual void* rowp(int row_idx) const override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                pixel_t* pointer = (pixel_t*)halide_image_t::data() +
                                   static_cast<off_t>(row_idx * rowp_stride);
                return static_cast<void*>(pointer);
            }
            
        protected:
            off_t rowp_stride;
    };
    
#undef CALCULATE_ROWP_STRIDE
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    template <typename T>
    class HalideFactory : public ImageFactory {
    
        public:
            using pixel_t  = T;
            using image_t  = HybridImage<pixel_t>;
            using unique_t = std::unique_ptr<Image>;
            using shared_t = std::shared_ptr<Image>;
        
        private:
            std::string factory_name;
            Halide::Type factory_type;
        
        public:
            
            HalideFactory()
                :factory_name("")
                ,factory_type(detail::halide_t<pixel_t>())
                {}
            HalideFactory(std::string const& n)
                :factory_name(n)
                ,factory_type(detail::halide_t<pixel_t>())
                {}
            
            virtual ~HalideFactory() {}
            
            Halide::Type type()                                 { return factory_type; }
            std::string const& name()                           { return factory_name; }
            std::string const& name(std::string const& nm)      { factory_name = nm;
                                                                  return factory_name; }
            
        protected:
            virtual unique_t create(int nbits,
                                    int xHEIGHT, int xWIDTH, int xDEPTH,
                                    int d3, int d4) override {
                return unique_t(new image_t(xWIDTH, xHEIGHT, xDEPTH));
            }
    };
    
#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
    
    namespace halide {
        
        static const Options empty_opts;
        
        template <typename T = byte> inline
        HybridImage<T> read(std::string const& filename,
                            Options const& opts = empty_opts) {
            HalideFactory<T> factory;
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory,
                                                         format->add_options(opts));
            HybridImage<T> image(dynamic_cast<HybridImage<T>&>(*output.get()));
            image.set_host_dirty();
            return image;
        }
        
        template <typename T = byte> inline
        std::unique_ptr<Image> unique(std::string const& filename,
                                      Options const& opts = empty_opts) {
            HalideFactory<T> factory;
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> out = format->read(input.get(), &factory,
                                                      format->add_options(opts));
            return out;
        }
        
        template <typename T = byte> inline
        ImageList read_multi(std::string const& filename,
                             Options const& opts = empty_opts) {
            HalideFactory<T> factory;
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            return format->read_multi(input.get(), &factory,
                                      format->add_options(opts));
        }
        
        template <typename T = byte> inline
        void write(HybridImage<T>& input, std::string const& filename,
                                          Options const& opts = empty_opts) {
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write(dynamic_cast<Image&>(input), output.get(),
                          format->add_options(opts));
        }
        
        inline void write_multi(ImageList& input, std::string const& filename,
                                                  Options const& opts = empty_opts) {
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write_multi(input, output.get(),
                                format->add_options(opts));
        }
        
        inline void write_multi_handle(ImageList& input, std::string const& filename,
                                                         Options const& opts = empty_opts) {
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<handle::sink> output(new handle::sink(filename));
            format->write_multi(input, output.get(),
                                format->add_options(opts));
        }
        
        template <typename FormatType, typename T = byte> inline
        std::string tmpwrite(HybridImage<T>& input,
                             Options const& opts = empty_opts) {
            using filesystem::TemporaryName;
            TemporaryName tn(FormatType::suffix(true), false); /// cleanup on scope exit
            std::string out = tn.do_not_destroy();
            std::unique_ptr<ImageFormat> format(new FormatType);
            std::unique_ptr<FileSink> output(new FileSink(out));
            format->write(dynamic_cast<Image&>(input), output.get(),
                          format->add_options(opts));
            return out;
        }
        
    }
    
}

#endif /// LIBIMREAD_HALIDE_HH_
