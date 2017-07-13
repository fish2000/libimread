/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_COREGRAPHICS_HH_
#define LIBIMREAD_COREGRAPHICS_HH_

#include <memory>
#include <cstdlib>
#include <libimread/corefoundation.hh>
#include <libimread/image.hh>
#include <libimread/metadata.hh>

#import  <CoreFoundation/CoreFoundation.h>
#import  <CoreVideo/CoreVideo.h>
#import  <ImageIO/ImageIO.h>

/// These macros, like much other stuff in this implementation TU,
/// are adapted from the always-handy-dandy Ray Wenderlich -- specifically, q.v.:
/// https://www.raywenderlich.com/69855/image-processing-in-ios-part-1-raw-bitmap-modification
/// http://www.modejong.com/blog/post3_pixel_binary_layout_w_premultiplied_alpha/index.html

#define UNCOMPAND(x)        ((x) & 0xFF)
#define R(x)                (UNCOMPAND(x))
#define G(x)                (UNCOMPAND(x >> 8 ))
#define B(x)                (UNCOMPAND(x >> 16))
#define A(x)                (UNCOMPAND(x >> 24))

// #define COMPAND(r, g, b, a) ((a & 0xFF) << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)
#define COMPAND(x)          (uint32_t)((x) & 0xFF)
#define RGBA(r, g, b, a)    (COMPAND(r)) | (COMPAND(g) << 8) | (COMPAND(b) << 16) | (COMPAND(a) << 24)

#define CG_FLOAT(x)     static_cast<CGFloat>(x)

/// kCVPixelFormatType_32ARGB
/// kCVPixelFormatType_32RGBA
/// k32ARGBPixelFormat

namespace cg {
    
    using im::Image;
    using im::ImageFactory;
    using im::Metadata;
    using im::detail::cfp_t;
    using im::detail::pixbuf_t;
    using cvpref_t = cfp_t<CVPixelBufferRef>;
    
    class PixelBuffer : public Image, public Metadata {
        
        public:
            
            PixelBuffer()
                :Image(), Metadata()
                {}
            
            PixelBuffer(PixelBuffer&& other) noexcept
                :Image(), Metadata(dynamic_cast<Metadata&&>(other).get_meta())
                ,width(other.width), height(other.height), channels(other.channels)
                ,bpc(8), bpp(bpc * channels), bpr(width * channels * (bpc / 8))
                ,internal(std::move(other.internal))
                ,pixels(std::move(other.pixels))
                {
                    if (internal.get() != nullptr) {
                        CVPixelBufferLockBaseAddress(internal.get(), 0);
                    }
                }
            
            PixelBuffer(int x, int y, int z, int w, std::string const& name="")
                :Image(), Metadata(name)
                ,width(x), height(y), channels(z)
                ,bpc(8), bpp(bpc * channels), bpr(width * channels * (bpc / 8))
                ,pixels(std::make_unique<uint32_t[]>(width * height))
                {
                    CVPixelBufferRef buffer = nullptr;
                    CVReturn result = CVPixelBufferCreateWithBytes(kCFAllocatorDefault,
                                                                   width, height,
                                                                   kCVPixelFormatType_32RGBA,
                                                                   static_cast<void*>(pixels.get()),
                                                                   bpr,
                                                                   nullptr, nullptr,
                                                                   nullptr,
                                                                   &buffer); /// kCVReturnSuccess
                    if (buffer != nullptr) {
                        internal.reset(buffer);
                        CVPixelBufferLockBaseAddress(internal.get(), 0);
                    }
                }
            
            PixelBuffer(int x, int y, int z, std::string const& name="")
                :Image(), Metadata(name)
                ,width(x), height(y), channels(z)
                ,bpc(8), bpp(bpc * channels), bpr(width * channels * (bpc / 8))
                ,pixels(std::make_unique<uint32_t[]>(width * height))
                {
                    CVPixelBufferRef buffer = nullptr;
                    CVReturn result = CVPixelBufferCreateWithBytes(kCFAllocatorDefault,
                                                                   width, height,
                                                                   kCVPixelFormatType_32RGBA,
                                                                   static_cast<void*>(pixels.get()),
                                                                   bpr,
                                                                   nullptr, nullptr,
                                                                   nullptr,
                                                                   &buffer); /// kCVReturnSuccess
                   if (buffer != nullptr) {
                       internal.reset(buffer);
                       CVPixelBufferLockBaseAddress(internal.get(), 0);
                   }
                }
            
            PixelBuffer(int x, int y, std::string const& name="")
                :Image(), Metadata(name)
                ,width(x), height(y)
                ,bpc(8), bpp(bpc * channels), bpr(width * channels * (bpc / 8))
                ,pixels(std::make_unique<uint32_t[]>(width * height))
                {
                    CVPixelBufferRef buffer = nullptr;
                    CVReturn result = CVPixelBufferCreateWithBytes(kCFAllocatorDefault,
                                                                   width, height,
                                                                   kCVPixelFormatType_32RGBA,
                                                                   static_cast<void*>(pixels.get()),
                                                                   bpr,
                                                                   nullptr, nullptr,
                                                                   nullptr,
                                                                   &buffer); /// kCVReturnSuccess
                   if (buffer != nullptr) {
                       internal.reset(buffer);
                       CVPixelBufferLockBaseAddress(internal.get(), 0);
                   }
                }
            
            PixelBuffer(int x, std::string const& name="")
                :Image(), Metadata(name)
                ,width(x), height(1)
                ,bpc(8), bpp(bpc * channels), bpr(width * channels * (bpc / 8))
                ,pixels(std::make_unique<uint32_t[]>(width * height))
                {
                    CVPixelBufferRef buffer = nullptr;
                    CVReturn result = CVPixelBufferCreateWithBytes(kCFAllocatorDefault,
                                                                   width, height,
                                                                   kCVPixelFormatType_32RGBA,
                                                                   static_cast<void*>(pixels.get()),
                                                                   bpr,
                                                                   nullptr, nullptr,
                                                                   nullptr,
                                                                   &buffer); /// kCVReturnSuccess
                   if (buffer != nullptr) {
                       internal.reset(buffer);
                       CVPixelBufferLockBaseAddress(internal.get(), 0);
                   }
                }
            
            virtual ~PixelBuffer() {
                if (internal.get() != nullptr) {
                    CVPixelBufferUnlockBaseAddress(internal.get(), 0);
                }
            }
            
            inline off_t rowp_stride() const {
                return static_cast<off_t>(
                    CVPixelBufferGetBytesPerRow(internal.get()));
            }
            
            virtual void* rowp(int r) const override {
                using pT = typename pixbuf_t::element_type;
                pT* host = pixels.get();
                host += off_t(r * rowp_stride());
                return static_cast<void*>(host);
            };
            
            virtual void* rowp() const override {
                return static_cast<void*>(pixels.get());
            }
            
            virtual int nbits() const override { return bpc; };
            virtual int nbytes() const override { return bpc / 8; };
            virtual int ndims() const override { return 3; };
            
            virtual int dim(int d) const override {
                switch (d) {
                    case 2:     return static_cast<int>(channels);
                    case 1:     return static_cast<int>(CVPixelBufferGetHeight(internal.get()));
                    case 0:     return static_cast<int>(CVPixelBufferGetWidth(internal.get()));
                    default:    return 1;
                }
            };
            
            virtual int stride(int s) const override {
                switch (s) {
                    case 2:     return static_cast<int>(1);
                    case 1:     return static_cast<int>(CVPixelBufferGetBytesPerRow(internal.get()));
                    case 0:     return static_cast<int>(channels);
                    default:    return 1;
                }
            };
            
            virtual bool is_signed() const override { return false; };
            virtual bool is_floating_point() const override { return false; };
            
        protected:
            
            std::size_t width = 0;          /// width in pixels;
            std::size_t height = 0;         /// height in pixels
            std::size_t channels = 4;       /// number of channels (R,G,B,A)
            std::size_t bpc = 8;            /// bits per component
            std::size_t bpp = 32;           /// bits per pixel
            std::size_t bpr = 0;            /// bytes per row
            cvpref_t internal{ nullptr };   /// CVPixelBufferRef pointer value
            pixbuf_t pixels{ nullptr };     /// CV pixel buffer base address
                                            /// … q.v. CVPixelBufferGetBaseAddress() sub.
    };
    
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
    
    class PixelFactory : public ImageFactory {
    
        public:
            using pixel_t  = uint32_t;
            using image_t  = PixelBuffer;
            using unique_t = std::unique_ptr<Image>;
        
        private:
            std::string factory_name;
        
        public:
            
            PixelFactory()
                :factory_name("")
                {}
            
            PixelFactory(std::string const& n)
                :factory_name(n)
                {}
            
            virtual ~PixelFactory() {}
            
            std::string const& name()                           { return factory_name; }
            std::string const& name(std::string const& nm)      { factory_name = nm; return name(); }
            
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
    
}


#endif /// LIBIMREAD_COREGRAPHICS_HH_