/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_VPP_H_
#define LIBIMREAD_VPP_H_

#include <cstring>
#include <memory>
#include <utility>
#include <cstdio>
#include <cstdint>

#include <type_traits>
#include <iod/sio.hh>
#include <iod/callable_traits.hh>
#include <iod/tuple_utils.hh>
#include <iod/utils.hh>
#include <iod/bind_method.hh>
#include <libimread/private/vpp_symbols.hh>

#include <vpp/vpp.hh>
#include <vpp/core/image2d.hh>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/base.hh>
#include <libimread/file.hh>
#include <libimread/formats.hh>

namespace im {
    
    template <typename V, unsigned N>
    using VppImage = ::vpp::imageNd<typename std::decay<V>::type, N>;
    using s::_pitch;
    
    template <typename pT, unsigned N = 3>
    class HybridImage : public VppImage<pT, N>, public Image {
        public:
            HybridImage() : Image() {}
            
            HybridImage(int x, int y, int z)
                :VppImage<pT, 3>(x, y, z, _pitch = x), Image()
                {}
            
            HybridImage(int x, int y)
                :VppImage<pT, 2>(x, y, _pitch = x), Image()
                {}
            
            HybridImage(int x)
                :VppImage<pT, 1>(x, _pitch = x), Image()
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
                return (int)N;
            }
            
            virtual int dim(int d) const override {
                return this->domain()->size(d);
            }
            
            inline off_t rowp_stride() const {
                return N < 3 ? 0 : off_t(this->pitch());
            }
            
            virtual void *rowp(int r) override {
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
    class VppFactory : public ImageFactory {
        private:
            std::string nm;
        
        public:
            typedef T pixel_type;
            
            VppFactory()
                :nm(std::string(""))
                {}
            VppFactory(const std::string &n)
                :nm(std::string(n))
                {}
            
            virtual ~VppFactory() {}
            
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
    };

#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
    
    namespace vpp {
        
        template <typename V = byte, unsigned N = 3>
        VppImage<V, N> read(const std::string &filename) {
            options_map opts;
            VppFactory<V> factory(filename);
            std::unique_ptr<ImageFormat> format(for_filename(filename));
            std::unique_ptr<FileSource> input(new FileSource(filename));
            std::unique_ptr<Image> output = format->read(input.get(), &factory, opts);
            HybridImage<V, N> image(dynamic_cast<HybridImage<V, N>&>(*output));
            return image;
        }
        
    }
    
}

#endif // LIBIMREAD_VPP_H_
