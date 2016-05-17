
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
    
    TEST_CASE("[gif-write] Read PNG files and write as individual GIF files",
              "[gif-write-individual-files]")
    {
        TemporaryDirectory td("write-individual-gifs");
        path individual_basedir(im::test::basedir);
        std::size_t idx = 0;
        std::vector<path> individual_sequence = individual_basedir.list("output_*.png");
        std::for_each(individual_sequence.begin(), individual_sequence.end(),
                  [&](path const& p) {
            path fullpath = individual_basedir/p;
            path newpath((td.dirpath/p).str() + ".gif");
            U8Image halim = im::halide::read(fullpath.str());
            im::halide::write(halim, newpath.str());
            CHECK(newpath.is_file());
            path copy = newpath.duplicate("/tmp/output-" + std::to_string(idx) + ".gif");
            CHECK(copy.is_file());
            idx++;
        });
    }
    
    TEST_CASE("[gif-write] Read PNG files and write as a single animated GIF file",
              "[gif-write-multi-animated]")
    {
        NamedTemporaryFile composite(".gif", FILESYSTEM_TEMP_FILENAME, false);
        path basedir(im::test::basedir);
        std::vector<path> sequence = basedir.list(std::regex("output_([0-9]+).png"));
        im::ImageList outlist;
        
        CHECK(composite.filepath.remove());
        std::for_each(sequence.begin(), sequence.end(),
                  [&](path const& p) {
            U8Image* halim = new U8Image(im::halide::read((basedir/p).str()));
            outlist.push_back(halim);
        });
        im::halide::write_multi(outlist, composite.str());
        CHECK(composite.filepath.is_file());
        WTF("COMPOSITE FILEPATH, DOGG:",
            composite.filepath.str(),
            composite.filepath.inode(),
            composite.filepath.filesize());
        path dupe(composite.filepath.duplicate("/private/tmp/output-animated.gif"));
        WTF("DUPE FILEPATH, DOGG:", dupe.str(), dupe.inode(), dupe.filesize());
        CHECK(dupe.is_file());
    }
    
}