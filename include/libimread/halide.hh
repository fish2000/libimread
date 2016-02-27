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

#include <Halide.h>

#include <libimread/libimread.hpp>
#include <libimread/private/buffer_t.h>
#include <libimread/errors.hh>
#include <libimread/fs.hh>
#include <libimread/image.hh>
#include <libimread/formats.hh>

namespace im {
    
    namespace detail {
        
        template <typename PixelType>
        struct for_type;
        
        #define DEFINE_TYPE_MAPPING(PixelType, StructType)                  \
        template <>                                                         \
        struct for_type<PixelType> {                                        \
            using type = PixelType;                                         \
            static Halide::Type get() { return Halide::StructType; }        \
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
        // DEFINE_TYPE_MAPPING(PixelType*,     Handle());
        
        template <typename PixelType> inline
        Halide::Type halide_t() { return for_type<std::remove_pointer_t<
                                                  std::decay_t<PixelType>>>::get(); }
        
    } /* namespace detail */
    
    template <typename T>
    using HalImage = Halide::Image<typename std::decay<T>::type>;
    using MetaImage = ImageWithMetadata;
    
    using Halide::Expr;
    using Halide::Buffer;
    using Halide::Realization;
    using Halide::Argument;
    using Halide::ExternFuncArgument;
    
    template <typename pT>
    class HybridImage : public HalImage<pT>, public Image, public MetaImage {
        public:
            using pixel_t = pT;
            using halide_image_t = HalImage<pT>;
            
            HybridImage()
                :halide_image_t(), Image(), MetaImage()
                {}
            
            HybridImage(int x, int y, int z, int w, const std::string& name="")
                :halide_image_t(x, y, z, w, name), Image(), MetaImage(name)
                {}
            
            HybridImage(int x, int y, int z, const std::string& name="")
                :halide_image_t(x, y, z, name), Image(), MetaImage(name)
                {}
            
            HybridImage(int x, int y, const std::string& name="")
                :halide_image_t(x, y, name), Image(), MetaImage(name)
                {}
            
            HybridImage(int x, const std::string& name="")
                :halide_image_t(x, name), Image(), MetaImage(name)
                {}
            
            HybridImage(const Buffer& buf)
                :halide_image_t(buf), Image(), MetaImage()
                {}
            HybridImage(const Realization& r)
                :halide_image_t(r), Image(), MetaImage()
                {}
            HybridImage(const buffer_t* b, const std::string& name="")
                :halide_image_t(b, name), Image(), MetaImage(name)
                {}
            
            using halide_image_t::operator();
            using halide_image_t::defined;
            using halide_image_t::dimensions;
            using halide_image_t::extent;
            using halide_image_t::stride;
            using halide_image_t::channels;
            using halide_image_t::data;
            using halide_image_t::buffer;
            
            virtual ~HybridImage() {}
            
            operator Buffer() const { return halide_image_t::buffer; }
            
            operator Argument() const {
                return Argument(halide_image_t::buffer);
            }
            
            operator ExternFuncArgument() const {
                return ExternFuncArgument(halide_image_t::buffer);
            }
            
            operator Expr() const {
                return (*this)(Halide::_);
            }
            
            Halide::Type type() const {
                return halide_image_t::buffer.type();
            }
            
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
            
            inline off_t rowp_stride() const {
                return halide_image_t::channels() == 1 ? 0 : off_t(halide_image_t::stride(1));
            }
            
            virtual void* rowp(int r) const override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                pT* host = (pT*)halide_image_t::data();
                host += off_t(r * rowp_stride());
                return static_cast<void*>(host);
            }
            
    };
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    template <typename T>
    class HalideFactory : public ImageFactory {
        private:
            std::string nm;
        
        public:
            using pixel_t = T;
            using image_t = HybridImage<T>;
            
            HalideFactory()
                :nm("")
                {}
            HalideFactory(const std::string& n)
                :nm(n)
                {}
            
            virtual ~HalideFactory() {}
            
            Halide::Type type() {
                return detail::halide_t<T>();
            }
            
            std::string& name() { return nm; }
            void name(std::string& nnm) { nm = nnm; }
            
        protected:
            virtual std::unique_ptr<Image> create(int nbits,
                                                  int xHEIGHT, int xWIDTH, int xDEPTH,
                                                  int d3, int d4) override {
                return std::unique_ptr<Image>(
                    new image_t(xWIDTH, xHEIGHT, xDEPTH));
            }
            
            virtual std::shared_ptr<Image> shared(int nbits,
                                                  int xHEIGHT, int xWIDTH, int xDEPTH,
                                                  int d3, int d4) override {
                return std::shared_ptr<Image>(
                    new image_t(xWIDTH, xHEIGHT, xDEPTH));
            }
    };
    
#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
    
    namespace halide {
        
        static const options_map halide_default_opts;
        
        template <typename T = byte> inline
        HybridImage<T> read(const std::string& filename,
                            const options_map& opts = halide_default_opts) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            options_map default_opts = format->add_options(opts);
            std::unique_ptr<Image> output = format->read(input.get(), &factory, default_opts);
            HybridImage<T> image(dynamic_cast<HybridImage<T>&>(*output.get()));
            image.set_host_dirty();
            return image;
        }
        
        template <typename T = byte> inline
        void write(HybridImage<T>& input, const std::string& filename,
                                          const options_map& opts = halide_default_opts) {
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            options_map default_opts = format->add_options(opts);
            format->write(dynamic_cast<Image&>(input), output.get(), default_opts);
        }
        
        inline void write_multi(ImageList& input, const std::string& filename,
                                                  const options_map& opts = halide_default_opts) {
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            options_map default_opts = format->add_options(opts);
            format->write_multi(input, output.get(), default_opts);
        }
        
        template <typename Format, typename T = byte> inline
        std::string tmpwrite(HybridImage<T>& input,
                             const options_map& opts = halide_default_opts) {
            im::fs::NamedTemporaryFile tf("." + Format::suffix(),       /// suffix
                                          FILESYSTEM_TEMP_FILENAME,     /// prefix (filename template)
                                          false); tf.remove();          /// cleanup on scope exit
            std::unique_ptr<ImageFormat> format(new Format);
            std::unique_ptr<FileSink> output(new FileSink(tf.str()));
            options_map default_opts = format->add_options(opts);
            format->write(dynamic_cast<Image&>(input), output.get(), default_opts);
            return tf.str();
        }
        
    }
    
}

#endif /// LIBIMREAD_HALIDE_HH_
