
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
    
    using pathvec_t = std::vector<path>;
    
    TEST_CASE("[gif-write] Read PNG files and write as individual GIF files",
              "[gif-write-individual-files]")
    {
        TemporaryDirectory td("write-individual-gifs");
        path basedir(im::test::basedir);
        const pathvec_t outglob = basedir.list("output_*.png");
        
        std::for_each(outglob.begin(), outglob.end(),
                  [&](path const& p) {
            path fullpath = basedir/p;
            path newpath((td.dirpath/p).str() + ".gif");
            U8Image halim = im::halide::read(fullpath.str());
            im::halide::write(halim, newpath.str());
            CHECK(newpath.is_file());
        });
    }
    
    TEST_CASE("[gif-write] Read PNG files and write as a single animated GIF file",
              "[gif-write-multi-animated]")
    {
        NamedTemporaryFile composite(".gif");
        ImageList outlist;
        path basedir(im::test::basedir);
        const pathvec_t sequence = basedir.list(std::regex("output_([0-9]+).png"));
        
        CHECK(composite.remove());
        std::for_each(sequence.begin(), sequence.end(),
                  [&](path const& p) {
            U8Image* halim = new U8Image(im::halide::read((basedir/p).str()));
            outlist.push_back(halim);
        });
        
        im::halide::write_multi(outlist, composite.str());
        CHECK(composite.filepath.is_file());
    }
    
}