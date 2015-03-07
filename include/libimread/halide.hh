// Copyright 2012-2014 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_HALIDE_H_
#define LIBIMREAD_HALIDE_H_

#include <cstring>
#include <memory>
#include <utility>
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
            
            virtual ~HybridImage() {}
            
            virtual int nbits() const override {
                /// elem_size is in BYTES, so:
                return sizeof(pT) * 8;
            }
            
            virtual int nbytes() const override {
                return sizeof(pT);
            }
            
            virtual int ndims() const override {
                return this->dimensions();
            }
            
            virtual int dim(int d) const override {
                return this->extent(d);
            }
            
            inline off_t rowp_stride() const {
                return this->channels() == 1 ? 0 : off_t(this->stride(1));
            }
            
            /*
            virtual void *at(int r) {
                return (void *)((pT *)this->data())[r*this->stride(1)];
            }
            virtual void *at(int x, int y=0, int z=0) {
                return (void *)((pT *)this->data())[x*this->stride(0) + y*this->stride(1) + z*this->stride(2)];
            }
            */
            
            virtual void *rowp(int r) override {
                /// WARNING: FREAKY POINTERMATH FOLLOWS
                pT *host = (pT *)this->data();
                host += off_t(r * rowp_stride());
                return static_cast<void *>(host);
            }
            
            template <typename T>
            T* rowp_as(const int r) {
                T *host = (T *)this->data();
                host += off_t(r * rowp_stride());
                return host;
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
                        xWIDTH, xHEIGHT, xDEPTH,
                        name()));
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

    namespace apple {
        
        static const options_map opts; /// not currently used when reading
        
        template <typename T = byte>
        Halide::Image<T> read(const std::string &filename) {
            HalideFactory<T> factory(filename);
            std::unique_ptr<ImageFormat> format(get_format("objc"));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            HybridImage<T> image(dynamic_cast<HybridImage<T>&>(*output));
            return image;
        }
        
    }
    
}

#endif // LIBIMREAD_HALIDE_H_
