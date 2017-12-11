
#define CATCH_CONFIG_FAST_COMPILE

#include <regex>
#include <vector>
#include <memory>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/imagelist.hh>
#include <libimread/halide.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::TemporaryName;
    using filesystem::TemporaryDirectory;
    using filesystem::path;
    using pathvec_t = std::vector<path>;
    using U8Image = im::HybridImage<uint8_t>;
    using im::ImageList;
    
    TEST_CASE("[tif-write-multi] Read PNG files and write as a single multi-image TIF file",
              "[tif-write-multi]")
    {
        TemporaryName composite(".tif");
        ImageList outlist;
        path basedir(im::test::basedir);
        const pathvec_t sequence = basedir.list(std::regex("output_([0-9]+).png"));
        
        /// build an ImageList
        std::for_each(sequence.begin(), sequence.end(),
                  [&](path const& p) {
            U8Image* halim = new U8Image(im::halide::read((basedir/p).str()));
            outlist.push_back(halim);
        });
        
        /// call write_multi()
        im::halide::write_multi(outlist, composite.str());
        CHECK(composite.pathname.is_file());
        
        /// try readback
        ImageList readback = im::halide::read_multi(composite.str());
        CHECK(readback.size() == sequence.size());
        CHECK(readback.size() == outlist.size());
        
        /// double-check individual Image and ImageList sizes
        int max = readback.size();
        outlist.compute_sizes();
        readback.compute_sizes();
        
        CHECK(outlist.width()  == readback.width());
        CHECK(outlist.height() == readback.height());
        CHECK(outlist.planes() == readback.planes());
        
        for (int idx = 0; idx < max; ++idx) {
            CHECK(outlist.at(idx)->width()  == readback.at(idx)->width());
            CHECK(outlist.at(idx)->height() == readback.at(idx)->height());
            CHECK(outlist.at(idx)->planes() == readback.at(idx)->planes());
        }
    }
    
    TEST_CASE("[tif-write-multi] Read PNG files and write as a single multi-image TIF file (with a filehandle)",
              "[tif-write-multi-filehandle]")
    {
        TemporaryName composite(".tif");
        ImageList outlist;
        path basedir(im::test::basedir);
        const pathvec_t sequence = basedir.list(std::regex("output_([0-9]+).png"));
        
        /// build an ImageList
        std::for_each(sequence.begin(), sequence.end(),
                  [&](path const& p) {
            U8Image* halim = new U8Image(im::halide::read((basedir/p).str()));
            outlist.push_back(halim);
        });
        
        /// call TIFFFormat::write_multi() via halide::write_multi_handle()
        im::halide::write_multi_handle(outlist, composite.str());
        CHECK(composite.pathname.is_file());
        
        /// try readback
        ImageList readback = im::halide::read_multi(composite.str());
        CHECK(readback.size() == sequence.size());
        CHECK(readback.size() == outlist.size());
        
        /// double-check individual Image and ImageList sizes
        int max = readback.size();
        outlist.compute_sizes();
        readback.compute_sizes();
        
        CHECK(outlist.width()  == readback.width());
        CHECK(outlist.height() == readback.height());
        CHECK(outlist.planes() == readback.planes());
        
        for (int idx = 0; idx < max; ++idx) {
            CHECK(outlist.at(idx)->width()  == readback.at(idx)->width());
            CHECK(outlist.at(idx)->height() == readback.at(idx)->height());
            CHECK(outlist.at(idx)->planes() == readback.at(idx)->planes());
        }
    }
}