
#include <vector>
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
    
    TEST_CASE("Read PNG files and write as individual GIF files", "[write-individual-gifs]") {
        path basedir(im::test::basedir);
        path tmpdir("/tmp");
        std::vector<path> sequence = basedir.list("output_*.png");
        std::for_each(sequence.begin(), sequence.end(), [&](path &p){
            path fullpath = basedir/p;
            path newpath((tmpdir/p).str() + ".gif");
            U8Image halim = im::halide::read(fullpath.str());
            im::halide::write(halim, newpath.str());
        });
    }
    
}