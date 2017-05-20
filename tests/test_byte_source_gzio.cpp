
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

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using im::byte;
    using im::bytevec_t;
    using im::FileSource;
    using GZSource = im::gzio::source;
    using GZSink = im::gzio::sink;
    
    using filesystem::TemporaryDirectory;
    using filesystem::path;
    using filesystem::detail::pathvec_t;
    
    TEST_CASE("[byte-source-gzio] GZIO byte_source compression roundtrip using PNG data",
              "[byte-source-gzio-gzio-byte_source-compression-roundtrip-png-data]")
    {
        path basedir(im::test::basedir);
        TemporaryDirectory td("test-gzio-pngs");
        const pathvec_t pngs = basedir.list("*.png");
        
        std::for_each(pngs.begin(), pngs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            path gzpath = td.dirpath/p + ".gz";
            std::unique_ptr<FileSource> source(new FileSource(imagepath));
            bytevec_t fulldata = source->full_data();
            bytevec_t readback;
            
            {
                /// nest scope to ensure GZSink gets rightly dumpstered
                std::unique_ptr<GZSink> gzoutput(new GZSink(gzpath));
                gzoutput->write(fulldata);
                CHECK(gzoutput->uncompressed_byte_size() == fulldata.size());
                // WTF("GZIO compression ratio: ",
                //     FF("\tFile size (compressed):   %u", gzoutput->size()),
                //     FF("\tData size (uncompressed): %u", gzoutput->uncompressed_byte_size()),
                //     FF("\tCompression ratio:        %f", gzoutput->compression_ratio()));
            }
            
            REQUIRE(gzpath.is_file());
            REQUIRE(gzpath.is_readable());
            
            {
                /// nest scope again -- same deal, but like, for GZSource readback
                /// NB. GZIO's `full_data()` is the pathological base class version
                std::unique_ptr<GZSource> gzinput(new GZSource(gzpath));
                readback = gzinput->full_data();
                CHECK(gzinput->uncompressed_byte_size() == readback.size());
                // WTF("GZIO compression ratio: ",
                //     FF("\tFile size (compressed):   %u", gzinput->size()),
                //     FF("\tData size (uncompressed): %u", gzinput->uncompressed_byte_size()),
                //     FF("\tCompression ratio:        %f", gzinput->compression_ratio()));
            }
            
            CHECK(readback.size() == fulldata.size());
            CHECK(std::equal(readback.begin(), readback.end(),
                             fulldata.begin(), fulldata.end(),
                             std::equal_to<byte>()));
        });
    }
    
    
    TEST_CASE("[byte-source-gzio] GZIO byte_source compression roundtrip using JPEG data",
              "[byte-source-gzio-gzio-byte_source-compression-roundtrip-jpeg-data]")
    {
        path basedir(im::test::basedir);
        TemporaryDirectory td("test-gzio-jpgs");
        const pathvec_t jpgs = basedir.list("*.jpg");
        
        std::for_each(jpgs.begin(), jpgs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            path gzpath = td.dirpath/p + ".gz";
            std::unique_ptr<FileSource> source(new FileSource(imagepath));
            bytevec_t fulldata = source->full_data();
            bytevec_t readback;
            
            {
                /// nest scope to ensure GZSink gets rightly dumpstered
                std::unique_ptr<GZSink> gzoutput(new GZSink(gzpath));
                gzoutput->write(fulldata);
                gzoutput->flush();
                CHECK(gzoutput->uncompressed_byte_size() == fulldata.size());
                WTF("GZIO compression ratio: ",
                    FF("\tFile size (compressed):   %u", gzoutput->size()),
                    FF("\tFile size (stat):         %u", gzpath.filesize()),
                    FF("\tData size (uncompressed): %u", gzoutput->uncompressed_byte_size()),
                    FF("\tCompression ratio:        %f", gzoutput->compression_ratio()));
            }
            
            REQUIRE(gzpath.is_file());
            REQUIRE(gzpath.is_readable());
            
            {
                /// nest scope again -- same deal, but like, for GZSource readback
                /// NB. GZIO's `full_data()` is the pathological base class version
                std::unique_ptr<GZSource> gzinput(new GZSource(gzpath));
                readback = gzinput->full_data();
                CHECK(gzinput->uncompressed_byte_size() == readback.size());
                WTF("GZIO compression ratio: ",
                    FF("\tFile size (compressed):   %u", gzinput->size()),
                    FF("\tFile size (stat):         %u", gzpath.filesize()),
                    FF("\tData size (uncompressed): %u", gzinput->uncompressed_byte_size()),
                    FF("\tCompression ratio:        %f", gzinput->compression_ratio()));
            }
            
            CHECK(readback.size() == fulldata.size());
            CHECK(std::equal(readback.begin(), readback.end(),
                             fulldata.begin(), fulldata.end(),
                             std::equal_to<byte>()));
        });
    }
    
}

