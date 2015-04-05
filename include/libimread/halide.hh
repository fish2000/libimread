// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HALIDE_H_
#define LIBIMREAD_HALIDE_H_

#include <iostream>
#include <memory>
#include <utility>
#include <cstring>
#include <cstdio>
#include <cstdint>

#include <Halide.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/base.hh>
#include <libimread/file.hh>
#include <libimread/formats.hh>
#include <libimread/tools.hh>

namespace im {
    
    template <typename T>
    using HalImage = Halide::Image<typename std::decay<T>::type>;
    using MetaImage = ImageWithMetadata;
    
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
            
            using HalImage<pT>::dimensions;
            using HalImage<pT>::extent;
            using HalImage<pT>::stride;
            using HalImage<pT>::channels;
            using HalImage<pT>::data;
            
            virtual ~HybridImage() {}
            
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
            std::unique_ptr<Image> create(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) {
                return std::unique_ptr<Image>(
                    new HybridImage<T>(
                        xWIDTH, xHEIGHT, xDEPTH));
            }
            
            std::shared_ptr<Image> shared(int nbits,
                                          int xHEIGHT, int xWIDTH, int xDEPTH,
                                          int d3, int d4) {
                return std::shared_ptr<Image>(
                    new HybridImage<T>(
                        xWIDTH, xHEIGHT, xDEPTH));
            }
    };

#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH

    namespace halide {
        
        static const options_map opts; /// not currently used when reading
        
        template <typename T = byte>
        Halide::Image<T> read(const std::string &filename) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            HybridImage<T> image(dynamic_cast<HybridImage<T>&>(*output));
            return image;
        }
        
    }
    
}

#endif // LIBIMREAD_HALIDE_H_
