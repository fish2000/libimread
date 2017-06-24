
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>

#include <libimread/file.hh>
#include <libimread/store.hh>
#include <libimread/corefoundation.hh>
#include <libimread/coregraphics.hh>
// #include <libkern/OSTypes.h>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using filesystem::TemporaryName;
    using filesystem::TemporaryDirectory;
    using filesystem::detail::stringvec_t;
    using im::FileSource;
    using im::FileSink;
    using im::detail::cfp_t;
    
    TEST_CASE("[cvpixelformat] Create a new pixel format description with a pixel format type",
              "[cvpixelformat-create-new-pixel-format-description-with-pixel-format-type]")
    {
        TemporaryDirectory td("test-cvpixelformat");                       /// the directory for all of this test’s genereated data
        OSType kFormatID = 0x88888888;
        
        cfp_t<CFDictionaryRef> pxfmt(const_cast<__CFDictionary *>(
            CVPixelFormatDescriptionCreateWithPixelFormatType(kCFAllocatorDefault, kFormatID)));
        
        // store::cfdict dict(CVPixelFormatDescriptionCreateWithPixelFormatType(kCFAllocatorDefault, kFormatID));
        // dict.set("kCVPixelFormatName", "YO DOGG");
        // CVPixelFormatDescriptionRegisterDescriptionWithPixelFormatType();
        
        // REQUIRE(pxfmt.get() != NULL);
        
        // store::cfdict dict(pxfmt.get());
        // WTF("PIXEL FORMAT DESCRIPTION DICT:", dict.to_string());
        
    }
    
}



/// CVPixelFormatDescriptionCreateWithPixelFormatType
/// CVPixelFormatDescriptionRegisterDescriptionWithPixelFormatType