
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/iterators.hh>
// #include <libimread/seekable.hh>
// #include <libimread/image.hh>
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
        
        std::for_each(pngs.begin(), pngs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            std::vector<byte> data;
            std::string pth = imagepath.str();
            std::unique_ptr<FileSource> source = std::make_unique<FileSource>(pth);
            std::for_each(std::begin(source.get()), std::end(source.get()), [&data](byte b) {
                data.push_back(b);
            });
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
        
        std::for_each(pngs.begin(), pngs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            std::string pth = imagepath.str();
            HandleSource source(pth);
            std::vector<byte> data;
            std::for_each(source.begin(), source.end(), [&data](byte b) {
                data.push_back(b);
            });
            std::vector<byte> fulldata(source.full_data());
            
            CHECK(data.size() == fulldata.size());
            CHECK(std::equal(data.begin(),     data.end(),
                             fulldata.begin(), fulldata.end(),
                             std::equal_to<byte>()));
            
        });
    }
    
}

