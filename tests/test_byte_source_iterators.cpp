
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/file.hh>
#include <libimread/filehandle.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using im::byte;
    using im::FileSource;
    using HandleSource = im::handle::source;
    using filesystem::path;
    
    TEST_CASE("[byte-source-iterators] Test FileSource iterators",
              "[byte-source-iterators-test-FileSource-iterators]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        
        std::for_each(pngs.begin(), pngs.end(), [&basedir](path const& p) {
            path imagepath = basedir/p;
            std::vector<byte> data;
            std::string pth = imagepath.str();
            std::unique_ptr<FileSource> source(new FileSource(pth));
            std::copy(std::begin(source.get()),
                      std::end(source.get()),
                      std::back_inserter(data));
            std::vector<byte> fulldata(source->full_data());
            
            CHECK(data.size() == fulldata.size());
            CHECK(std::equal(data.begin(),     data.end(),
                             fulldata.begin(), fulldata.end(),
                             std::equal_to<byte>()));
        });
    }
    
    
    TEST_CASE("[byte-source-iterators] Test im::handle::source iterators",
              "[byte-source-iterators-test-handle-source-iterators]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        
        std::for_each(pngs.begin(), pngs.end(), [&basedir](path const& p) {
            path imagepath = basedir/p;
            std::vector<byte> data;
            std::string pth = imagepath.str();
            HandleSource source(pth);
            std::copy(source.begin(), source.end(),
                      std::back_inserter(data));
            std::vector<byte> fulldata(source.full_data());
            
            CHECK(data.size() == fulldata.size());
            CHECK(std::equal(data.begin(),     data.end(),
                             fulldata.begin(), fulldata.end(),
                             std::equal_to<byte>()));
        });
    }
    
    TEST_CASE("[byte-source-iterators] Search for EXIF tag markers",
              "[byte-source-iterators-search-for-exif-tag-markers]")
    {
        path basedir(im::test::basedir);
        const std::string marker = "Exif\0\0";
        const std::vector<path> jpgs = basedir.list("*.jpg");
        
        std::for_each(jpgs.begin(), jpgs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            std::vector<byte> data;
            std::string pth = imagepath.str();
            FileSource source(pth);
            auto result = std::search(source.begin(), source.end(),
                                      marker.begin(), marker.end());
            bool has_exif = result == source.end();
            if (has_exif) {
                WTF("EXIF marker found at offset:", result - source.begin());
            } else {
                WTF("EXIF marker not found");
            }
            CHECK(has_exif == (result == source.end()));
        });
    }
    
}

