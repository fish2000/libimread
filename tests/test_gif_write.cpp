
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
    
    TemporaryDirectory td("write-individual-gifs-XXXXX");
    path basedir(im::test::basedir);
    std::vector<path> sequence = basedir.list("output_*.png");
    
    TEST_CASE("Read PNG files and write as individual GIF files", "[write-individual-gifs]") {
        std::for_each(sequence.begin(), sequence.end(), [&](path &p) {
            path fullpath = basedir/p;
            path newpath((td.dirpath/p).str() + ".gif");
            U8Image halim = im::halide::read(fullpath.str());
            im::halide::write(halim, newpath.str());
        });
    }
    
    TEST_CASE("Read PNG files and write as a single animated GIF file", "[write-animated-gif]") {
        path newpath(td.dirpath/"output_animated.gif");
        im::ImageList outlist;
        std::for_each(sequence.begin(), sequence.end(), [&](path &p) {
            path fullpath = p.make_absolute();
            U8Image halim = im::halide::read(fullpath.str());
            outlist.push_back(&halim);
        });
        im::halide::write_multi(outlist, newpath.str());
    }
    
}