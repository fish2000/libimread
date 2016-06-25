
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/exif.hh>
#include <libimread/file.hh>
#include <libimread/filehandle.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using im::byte;
    using im::FileSource;
    using HandleSource = im::handle::source;
    using im::byte_iterator;
    using filesystem::path;
    using easyexif::EXIFInfo;
    
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
    
    template <typename Iterator>
    uint16_t parse_size(Iterator it) {
        Iterator siz0 = std::next(it, 2);
        Iterator siz1 = std::next(it, 3);
        return (static_cast<uint16_t>(*siz0) << 8) | *siz1;
    }
    
    TEST_CASE("[byte-source-iterators] Search for EXIF tag markers",
              "[byte-source-iterators-search-for-exif-tag-markers]")
    {
        path basedir(im::test::basedir);
        const std::vector<byte> marker{ 0xFF, 0xE1 };
        const std::vector<path> jpgs = basedir.list("*.jpg");
        
        std::for_each(jpgs.begin(), jpgs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            std::vector<byte> data;
            std::string pth = imagepath.str();
            FileSource source(pth);
            byte_iterator result = std::search(source.begin(), source.end(),
                                               marker.begin(), marker.end());
            bool has_exif = result != source.end();
            if (has_exif) {
                uint16_t size = parse_size(result);
                std::advance(result, 4);
                std::copy(result, result + size,
                          std::back_inserter(data));
                
                // char m[6];
                // std::memcpy(m, &result, sizeof(m));
                // WTF("EXIF marker found at offset:", std::size_t(result),
                //     "with size:", size,
                //     "within size:", source.size(),
                //  FF("with value: %s", m));
                
                EXIFInfo exif;
                int parseResult = exif.parseFromEXIFSegment(&data[0], data.size());
                CHECK(parseResult == PARSE_EXIF_SUCCESS);
                
                WTF("EXIF data extracted:",
                    FF("\tImage Description: %s",   exif.ImageDescription.c_str()),
                    FF("\tSoftware Used: %s",       exif.Software.c_str()),
                    FF("\tMake: %s",                exif.Make.c_str()),
                    FF("\tModel: %s",               exif.Model.c_str()),
                    FF("\tCopyright: %s",           exif.Copyright.c_str()),
                    FF("\tImage Size: %ux%u",       exif.ImageWidth, exif.ImageHeight)
                );
                
            } else {
                // WTF("EXIF marker not found");
            }
        });
    }
    
}

