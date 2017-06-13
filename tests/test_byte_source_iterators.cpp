
#include <array>
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
    using im::bytevec_t;
    using im::FileSource;
    using HandleSource = im::handle::source;
    using im::byte_iterator;
    using filesystem::path;
    using pathvec_t = std::vector<path>;
    using easyexif::EXIFInfo;
    
    TEST_CASE("[byte-source-iterators] Test FileSource iterators",
              "[byte-source-iterators-test-FileSource-iterators]")
    {
        path basedir(im::test::basedir);
        const pathvec_t pngs = basedir.list("*.png", true); /// full_paths=true
        
        std::for_each(pngs.begin(), pngs.end(), [](path const& p) {
            bytevec_t data;
            std::unique_ptr<FileSource> source(new FileSource(p));
            std::copy(std::begin(source.get()),
                      std::end(source.get()),
                      std::back_inserter(data));
            bytevec_t fulldata(source->full_data());
            
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
        const pathvec_t pngs = basedir.list("*.png", true); /// full_paths=true
        
        std::for_each(pngs.begin(), pngs.end(), [](path const& p) {
            bytevec_t data;
            HandleSource source(p);
            std::copy(source.begin(),
                      source.end(),
                      std::back_inserter(data));
            bytevec_t fulldata(source.full_data());
            
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
        const std::array<byte, 2> marker{ 0xFF, 0xE1 };
        const pathvec_t jpgs = basedir.list("*.jpg", true); /// full_paths=true
        // const std::vector<path> pngs = basedir.list("*.png", true);
        // const std::vector<path> tifs = basedir.list("*.tif", true);
        
        auto exif_extractor = [&marker](path const& p) {
            FileSource source(p);
            bytevec_t exif_bytes;
            byte_iterator result = std::search(source.begin(), source.end(),
                                               marker.begin(), marker.end());
            bool has_exif = result != source.end();
            if (has_exif) {
                uint16_t size = parse_size(result);
                std::advance(result, 4);
                std::copy(result, result + size,
                          std::back_inserter(exif_bytes));
                
                // char m[6];
                // std::memcpy(m, &result, sizeof(m));
                // WTF("EXIF marker found at offset:", std::size_t(result),
                //     "with size:", size,
                //     "within size:", source.size(),
                //  FF("with value: %s", m));
                
                EXIFInfo exif;
                CHECK(exif.parseFromEXIFSegment(exif_bytes.data(),
                                                exif_bytes.size()) == PARSE_EXIF_SUCCESS);
                
                // WTF("EXIF data extracted:",
                //     FF("\tImage Description: %s",   exif.ImageDescription.c_str()),
                //     FF("\tSoftware Used: %s",       exif.Software.c_str()),
                //     FF("\tMake: %s",                exif.Make.c_str()),
                //     FF("\tModel: %s",               exif.Model.c_str()),
                //     FF("\tCopyright: %s",           exif.Copyright.c_str()),
                //     FF("\tImage Size: %ux%u",       exif.ImageWidth, exif.ImageHeight)
                // );
            }
        };
        
        std::for_each(jpgs.begin(), jpgs.end(), exif_extractor);
        // std::for_each(pngs.begin(), pngs.end(), exif_extractor);
        // std::for_each(tifs.begin(), tifs.end(), exif_extractor);
    }
    
}

