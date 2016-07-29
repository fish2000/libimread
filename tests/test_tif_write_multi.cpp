
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
    
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    using filesystem::path;
    using U8Image = im::HybridImage<uint8_t>;
    using im::ImageList;
    
    TEST_CASE("[tif-write-multi] Read PNG files and write as a single multi-image TIF file",
              "[tif-write-multi]")
    {
        NamedTemporaryFile composite(".tif");
        ImageList outlist;
        path basedir(im::test::basedir);
        const std::vector<path> sequence = basedir.list(std::regex("output_([0-9]+).png"));
        
        /// build an ImageList
        CHECK(composite.remove());
        std::for_each(sequence.begin(), sequence.end(),
                  [&](path const& p) {
            U8Image* halim = new U8Image(im::halide::read((basedir/p).str()));
            outlist.push_back(halim);
        });
        
        /// call write_multi()
        im::halide::write_multi(outlist, composite.str());
        CHECK(composite.filepath.is_file());
        
        /// try readback
        ImageList readback = im::halide::read_multi(composite.str());
        CHECK(readback.size() == sequence.size());
        CHECK(readback.size() == outlist.size());
    }
    
}