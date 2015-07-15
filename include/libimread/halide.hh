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
// #include <libimread/base.hh>
#include <libimread/file.hh>
#include <libimread/image.hh>
#include <libimread/formats.hh>
#include <libimread/tools.hh>

namespace im {
    
    template <typename T>
    using HalImage = Halide::Image<typename std::decay<T>::type>;
    using MetaImage = ImageWithMetadata;
    
    using Halide::Expr;
    using Halide::Buffer;
    using Halide::Realization;
    using Halide::Argument;
    using Halide::ExternFuncArgument;
    
    //using HalUnderscore = Halide::_; /// OMG GUYS THATS MY UNDERSCORE
    
    template <typename pT>
    class HybridImage : public HalImage<pT>, public Image, public MetaImage {
        public:
            HybridImage()
                :HalImage<pT>(), Image(), MetaImage()
                {}
            
            HybridImage(int x, int y, int z, int w, const std::string &name="")
                :HalImage<pT>(x, y, z, w, name), Image(), MetaImage(name)
                {}
            
            HybridImage(int x, int y, int z, const std::string &name="")
                :HalImage<pT>(x, y, z, name), Image(), MetaImage(name)
                {}
            
            HybridImage(int x, int y, const std::string &name="")
                :HalImage<pT>(x, y, name), Image(), MetaImage(name)
                {}
            
            HybridImage(int x, const std::string &name="")
                :HalImage<pT>(x, name), Image(), MetaImage(name)
                {}
            
            HybridImage(const Buffer &buf)
                :HalImage<pT>(buf), Image(), MetaImage()
                {}
            HybridImage(const Realization &r)
                :HalImage<pT>(r), Image(), MetaImage()
                {}
            HybridImage(const buffer_t *b, const std::string &name="")
                :HalImage<pT>(b, name), Image(), MetaImage(name)
                {}
            
            using HalImage<pT>::operator();
            using HalImage<pT>::defined;
            using HalImage<pT>::dimensions;
            using HalImage<pT>::extent;
            using HalImage<pT>::stride;
            using HalImage<pT>::channels;
            using HalImage<pT>::data;
            using HalImage<pT>::buffer;
            
            virtual ~HybridImage() {}
            
            operator Buffer() const { return HalImage<pT>::buffer; }
            
            operator Argument() const {
                return Argument(HalImage<pT>::buffer);
            }
            
            operator ExternFuncArgument() const {
                return ExternFuncArgument(HalImage<pT>::buffer);
            }
            
            operator Expr() const {
                return (*this)(Halide::_);
            }
            
            virtual int nbits() const override {
                /// elem_size is in BYTES, so:
                return sizeof(pT) * 8;
            }
            
            virtual int nbytes() const override {
                return sizeof(pT);
            }
            
            virtual int ndims() const override {
                return HalImage<pT>::dimensions();
            }
            
            virtual int dim(int d) const override {
                return HalImage<pT>::extent(d);
            }
            
            virtual int stride(int s) const override {
                return HalImage<pT>::stride(s);
            }
            
            inline off_t rowp_stride() const {
                return HalImage<pT>::channels() == 1 ? 0 : off_t(HalImage<pT>::stride(1));
            }
            
            virtual void *rowp(int r) override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                pT *host = (pT *)HalImage<pT>::data();
                host += off_t(r * rowp_stride());
                return static_cast<void *>(host);
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
            typedef T pixel_type;
            
            HalideFactory()
                :nm(std::string(""))
                {}
            HalideFactory(const std::string &n)
                :nm(std::string(n))
                {}
            
            virtual ~HalideFactory() {}
            
            std::string &name() { return nm; }
            void name(std::string &nnm) { nm = nnm; }
            
        protected:
            virtual std::unique_ptr<Image> create(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) override {
                return std::unique_ptr<Image>(
                    new HybridImage<T>(
                        xWIDTH, xHEIGHT, xDEPTH));
            }
            
            virtual std::shared_ptr<Image> shared(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) override {
                return std::shared_ptr<Image>(
                    new HybridImage<T>(
                        xWIDTH, xHEIGHT, xDEPTH));
            }
    };

#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH

    namespace halide {
        
        template <typename T = byte>
        HybridImage<T> read(const std::string &filename) {
            options_map opts;
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            HybridImage<T> image(dynamic_cast<HybridImage<T>&>(*output));
            image.set_host_dirty();
            return image;
        }
        
        using Halide::Func;
        using Halide::Var;
        using Halide::Target;
        using Halide::UInt;
        
        template <typename T = byte>
        void write(HybridImage<T> &input, const std::string &filename) {
            if (input.dim(2) > 3) { return; }
            options_map opts;
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write(dynamic_cast<Image&>(input), output.get(), opts);
        }
        
        template <typename T = byte>
        void write_multi(std::vector<HybridImage<T>> &input, const std::string &filename) {
            options_map opts;
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSink> output(new FileSink(filename));
            format->write_multi(dynamic_cast<std::vector<Image>&>(input), output.get(), opts);
        }
        
    }
    
}

#endif /// LIBIMREAD_HALIDE_HH_
