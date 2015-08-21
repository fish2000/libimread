
#import <Foundation/Foundation.h>
#import <SSZipArchive.h>

#include <libimread/libimread.hpp>
#include <libimread/ext/categories/NSString+STL.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/fs.hh>

#include "include/catch.hpp"

namespace {
    
    im::fs::TemporaryDirectory td("test-libsszip-XXXXX");
    
    TEST_CASE("[libssip] Create zip archive with contents of directory",
              "[libsszip-create-zip-with-directory]")
    {
        im::fs::path zpth = td.dirpath/"test-directory.zip";
        NSString *zipPath = [NSString stringWithSTLString:zpth.str()];
        NSString *dirPath = @"/Users/fish/Dropbox/libimread/tests/data";
        BOOL created = [SSZipArchive createZipFileAtPath:zipPath
                                 withContentsOfDirectory:dirPath];
        REQUIRE(created == YES);
        bool removed = zpth.remove();
        REQUIRE(removed == true);
    }
    
}

