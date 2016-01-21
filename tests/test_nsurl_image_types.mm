
#include <array>
#include <string>
#include <algorithm>
#include <unordered_map>

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSURL+IM.hh>
#include <libimread/fs.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using im::fs::path;
    using im::fs::switchdir;
    using im::fs::resolver;
    using im::fs::NamedTemporaryFile;
    using im::fs::TemporaryDirectory;
    using NSTYPE = NSBitmapImageFileType;
    
    std::array<NSTYPE, 6> types = {
        NSTIFFFileType, NSBMPFileType, NSGIFFileType,
        NSJPEGFileType, NSPNGFileType, NSJPEG2000FileType
    };
    
    std::array<std::string, 6> suffixes = {
        "tiff",     "bmp",      "gif",
        "jpg",      "png",      "jp2"
    };
    
    TEST_CASE("[nsurl-image-types] Check suffixes and image type functions",
              "[nsurl-check-suffixes-image-type-functions]")
    {
        
        std::unordered_map<NSTYPE, std::string> typemap;
        
        std::transform(types.begin(), types.end(),
                       suffixes.begin(),
                       std::inserter(typemap, typemap.begin()),
                       [](NSTYPE type, std::string const& sufx) {
            CHECK(type == objc::image::filetype(sufx));
            CHECK(sufx == objc::image::suffix(type));
            return std::make_pair(type, sufx);
        });
    }
    
    
    TEST_CASE("[nsurl-image-types] Check `isImage` and `imageFileType` NSURL category methods",
              "[nsurl-check-isimage-imagefiletype-category-methods]")
    {
        path basedir(im::test::basedir);
        std::vector<path> files = basedir.list("*.*", true); /// full_paths=true
        
        @autoreleasepool {
            
            std::for_each(files.begin(), files.end(),
                          [&](path const& p) {
                if (!p.is_file()) { return; }
                NSURL* urlpath = [NSURL fileURLWithFilesystemPath:p];
                CHECK([urlpath isImage] == YES);
                CHECK([urlpath imageFileType] == objc::image::filetype(p.extension()));
                //CHECK(objc::image::suffix([urlpath imageFileType]) == p.extension());
            });
            
        };
    }
    
}

