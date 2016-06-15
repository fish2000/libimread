
#include <array>
#include <string>
#include <algorithm>
#include <unordered_map>

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSURL+IM.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/directory.h>
#include <libimread/ext/filesystem/resolver.h>
#include <libimread/ext/filesystem/temporary.h>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using filesystem::switchdir;
    using filesystem::resolver;
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    using NSTYPE = NSBitmapImageFileType;
    
    std::array<NSTYPE, 7> types = {
        NSTIFFFileType, NSBMPFileType, NSGIFFileType,
        NSJPEGFileType, NSPNGFileType, NSJPEG2000FileType,
        AXPVRFileType
    };
    
    std::array<std::string, 7> suffixes = {
        "tiff",     "bmp",      "gif",
        "jpg",      "png",      "jp2",
        "pvr"
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

