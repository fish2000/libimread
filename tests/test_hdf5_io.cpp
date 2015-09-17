
#include <algorithm>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/fs.hh>
#include <libimread/IO/hdf5.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    
    TEST_CASE("[hdf5-io] Read PNG and JPEG files and write as individual HDF5 binary store files",
              "[hdf5-read-png-jpeg-write-individual-hdf5-files]")
    {
        im::fs::TemporaryDirectory td("test-hdf5-io-XXXXX");
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        const std::vector<path> jpgs = basedir.list("*.jpg");
        
        std::for_each(pngs.begin(), pngs.end(), [&basedir, &td](const path &p) {
            auto png = im::halide::read(basedir/p);
            path np = td.dirpath/p;
            std::string npext = np.str() + ".hdf5";
            im::halide::write(png, npext);
        });
        
        std::for_each(jpgs.begin(), jpgs.end(), [&basedir, &td](const path &p) {
            auto jpg = im::halide::read(basedir/p);
            path np = td.dirpath/p;
            std::string npext = np.str() + ".hdf5";
            im::halide::write(jpg, npext);
        });
        
        const std::vector<path> hdfs = td.dirpath.list("*.hdf5");
        
        std::for_each(hdfs.begin(), hdfs.end(), [&basedir, &td](const path &p) {
            path np = td.dirpath/p;
            auto hdf = im::halide::read(np);
            REQUIRE(hdf.dim(0) > 0);
            REQUIRE(hdf.dim(1) > 0);
        });
        
    }
  
};