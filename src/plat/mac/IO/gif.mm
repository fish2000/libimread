// Copyright 2017 Alexander Bohn <fish2000@gmail.com>
// License: MIT (see COPYING.MIT file)

#include <libimread/IO/gif.hh>
#include <libimread/pixels.hh>
#include <libimread/coregraphics.hh>

namespace im {
    
    std::unique_ptr<Image> GIFFormat::read(byte_source* src,
                                           ImageFactory* factory,
                                           options_map const& opts) {
        
        /// byte_source* src -> CFDataRef
        /// N.B. using kCFAllocatorNull as the fourth parameter ensures that no deallocator
        /// will be invoked upon the CFDataRef’s destruction (which is what we want, as we’re)
        /// only referencing the bytes from the byte_source* `src`; these could be the backing
        /// store of a std::vector, an mmapped file descriptor, a transient memory buffer, or
        /// whatever the fuck other shit, by design I would not know, actually
        detail::cfp_t<CFDataRef> sourcedata(const_cast<__CFData *>(
                CFDataCreateWithBytesNoCopy(kCFAllocatorDefault,
                                            src->data(),
                                            src->size(),
                                            kCFAllocatorNull)));
        
        // detail::cfp_t<CGDataProviderRef> provider(
        //            CGDataProviderCreateWithCFData(sourcedata.get()));
        
        /// CFDataRef -> CGImageSourceRef
        detail::cfp_t<CGImageSourceRef> source(
                   CGImageSourceCreateWithData(sourcedata.get(),
                                               nullptr));
        
        detail::cfp_t<CFDictionaryRef> options(nullptr);                            /// empty dictionary ref ptr
        
        {
            /// Deal with metadata:
            // detail::cfp_t<CFDictionaryRef> metadata(const_cast<__CFDictionary *>(
            //      CGImageSourceCopyPropertiesAtIndex(source.get(),
            //                                         CF_IDX(0),
            //                                         nullptr)));
            
            /// Deal with options, as passed to CGImageSourceCreateImageAtIndex(…):
            const void* keys[]      = { kCGImageSourceShouldCache,
                                        kCGImageSourceShouldAllowFloat,
                                        kCGImageSourceTypeIdentifierHint };
            
            const void* values[]    = { kCFBooleanTrue,                             /// YES CACHING;
                                        kCFBooleanFalse,                            /// NO FLOATS;
                                        kUTTypeGIF };                               /// ITS A GIF, DOGG
            
            /// CFDictionaryCreate() copies CFTypes from `keys` and `values`
            /// N.B. consider using kCFCopyStringDictionaryKeyCallBacks
            options.reset(const_cast<__CFDictionary *>(
                                    CFDictionaryCreate(kCFAllocatorDefault,
                                                       keys, values, CF_IDX(3),
                                                      &kCFTypeDictionaryKeyCallBacks,
                                                      &kCFTypeDictionaryValueCallBacks)));
        }
        
        /// CGImageSourceRef -> CGImageRef
        detail::cfp_t<CGImageRef> image(
        CGImageSourceCreateImageAtIndex(source.get(),                               /// CGImageSourceRef
                                        CF_IDX(0),                                  /// new image index
                                        options.get()));                            /// CFDictionaryRef
        
        /// CGImageRef -> CGColorSpaceRef: image (source) colorspace
        detail::cfp_t<CGColorSpaceRef> colorspace(
                             CGImageGetColorSpace(image.get()));                    /// Image (source) colorspace
        
        /// Image bitmap dimensions:
        std::size_t width = CGImageGetWidth(image.get()),
                   height = CGImageGetHeight(image.get()),
               components = CGColorSpaceGetNumberOfComponents(colorspace.get()),    /// colorspace component count
                      bpc = CGImageGetBitsPerComponent(image.get()),                /// bits per component
                      bpp = CGImageGetBitsPerPixel(image.get()) / bpc,              /// BYTES per pixel (not bits)
                      bpr = bpp * width;                                            /// bytes per row
      
        /// CGRect encompassing image bounds:
        CGRect bounds{ 0, 0, CG_FLOAT(width),
                             CG_FLOAT(height) };
        
        /// pixelbuffer: uint32_t[width * height]
        detail::pixbuf_t pixbuf = std::make_unique<uint32_t[]>(width * height);     /// pixels are 32-bit, for 4x 8-bit components
        
        /// CGColorSpaceRef: destination colorspace
        detail::cfp_t<CGColorSpaceRef> deviceRGB(CGColorSpaceCreateDeviceRGB());    /// Device (destination) colorspace
        
        /// CGImageRef + pixelbuffer -> CGContextRef
        // kCGImageAlphaPremultipliedLast |
        // kCGBitmapByteOrder32Big
        detail::cfp_t<CGContextRef> context(
                      CGBitmapContextCreate(pixbuf.get(),
                                            width, height,
                                            bpc, bpr,
                                            deviceRGB.get(),
                                            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big));
        
        /// Render image to bitmap context:
        CGContextDrawImage(context.get(), bounds, image.get());
        
        /// create new im::Image instance using factory pointer:
        // std::unique_ptr<Image> output = factory->create(bpc,                        /// bit depth (should be 8)
        //                                                 height,                     /// image height
        //                                                 width,                      /// image width
        //                                                 bpp);                       /// channel count (likely 4 for RGBA)
        
        std::unique_ptr<Image> output = factory->create(8, height, width, 3);
        
        /// Temporary values and pointers
        uint32_t* currentpixel = pixbuf.get();
        uint32_t  c_stride = (bpp == 1) ? 0 : output->stride(2);
        uint32_t  x, y, compand;
        byte*     destPtr;
        
        /// Read from pixel buffer, copy to newly created im::Image:
        for (y = 0; y < height; ++y) {
            destPtr = output->rowp_as<byte>(y);
            for (x = 0; x < width; ++x) {
                compand = *currentpixel;
                // pix::convert(R(compand), destPtr[0]);                               /// 0 * c_stride
                // pix::convert(G(compand), destPtr[c_stride]);                        /// 1 * c_stride
                // pix::convert(B(compand), destPtr[2*c_stride]);
                // pix::convert(A(compand), destPtr[3*c_stride]);
                
                destPtr[0] = R(compand);
                destPtr[c_stride] = G(compand);
                destPtr[2*c_stride] = B(compand);
                
                currentpixel++;
                destPtr++;
            }
        }
        
        /// return unique_ptr to new im::Image instance:
        return output;
    }
    
}