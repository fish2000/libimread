
#include <vector>
#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/base.hh>
#include <libimread/fs.hh>
#include <libimread/halide.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

#define D(pth) "/Users/fish/Dropbox/libimread/tests/data/" pth
#define T(pth) "/tmp/" pth

namespace {
    
    using namespace Halide;
    using namespace im::fs;
    using U8Image = im::HybridImage<uint8_t>;
    
    TEST_CASE("[gif-write] Read PNG files and write as individual GIF files", "[gif-write-individual-files]") {
        TemporaryDirectory td("write-individual-gifs");
        path individual_basedir(im::test::basedir);
        std::vector<path> individual_sequence = individual_basedir.list("output_*.png");
        std::for_each(individual_sequence.begin(), individual_sequence.end(), [&](path &p) {
            path fullpath = individual_basedir/p;
            path newpath((td.dirpath/p).str() + ".gif");
            U8Image halim = im::halide::read(fullpath.str());
            im::halide::write(halim, newpath.str());
            CHECK(newpath.is_file());
        });
    }
    
    TEST_CASE("[gif-write] Read PNG files and write as a single animated GIF file", "[gif-write-multi-animated]") {
        TemporaryDirectory td("write-animated-gif");
        path basedir(im::test::basedir);
        std::vector<path> sequence = basedir.list(std::regex("output_([0-9]+).png"));
        path newpath(td.dirpath/"output_animated.gif");
        im::ImageList outlist;
        std::for_each(sequence.begin(), sequence.end(), [&](path &p) {
            U8Image halim = im::halide::read((basedir/p).str());
            outlist.push_back(&halim);
        });
        im::halide::write_multi(outlist, newpath.str());
        outlist.release(); /// outlist will otherwise try to delete pointers to stack-allocated images
        CHECK(newpath.is_file());
    }
    
}