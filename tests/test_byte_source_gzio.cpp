
#include <array>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/gzio.hh>
#include <libimread/file.hh>
#include <libimread/filehandle.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using im::byte;
    using im::FileSource;
    using HandleSource = im::handle::source;
    using GZSource = im::gzio::source;
    using GZSink = im::gzio::sink;
    
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    using filesystem::path;
    
    TEST_CASE("[byte-source-gzio] Basic GZIO byte_source sanity check",
              "[byte-source-gzio-basic-gzio-byte_source-sanity-check]")
    {
        path basedir(im::test::basedir);
        TemporaryDirectory td("test-byte-source-gzio");
        const std::vector<path> pngs = basedir.list("*.png");
        
        std::for_each(pngs.begin(), pngs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            path gzpath = td.dirpath/p + ".gz";
            std::string pth = imagepath.str();
            std::string ttp = gzpath.str();
            std::unique_ptr<FileSource> source(new FileSource(pth));
            std::vector<byte> fulldata = source->full_data();
            std::vector<byte> readback;
            
            {
                /// nest scope to ensure GZSink gets rightly dumpstered
                std::unique_ptr<GZSink> gzoutput(new GZSink(ttp));
                gzoutput->write(fulldata);
            }
            
            REQUIRE(gzpath.is_file());
            
            {
                /// nest scope again -- same deal, but like, for GZSource readback
                /// NB. GZIO's `full_data()` is the pathological base class version
                std::unique_ptr<GZSource> gzinput(new GZSource(ttp));
                readback = gzinput->full_data();
            }
            
            CHECK(readback.size() == fulldata.size());
            CHECK(std::equal(readback.begin(), readback.end(),
                             fulldata.begin(), fulldata.end(),
                             std::equal_to<byte>()));
        });
    }
    
}

