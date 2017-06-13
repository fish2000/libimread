
#include <algorithm>
#include <vector>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/ext/filesystem/mode.h>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/directory.h>
#include <libimread/ext/filesystem/resolver.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/file.hh>
#include <libimread/filehandle.hh>
#include <libimread/IO/hdf5.hh>
#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using filesystem::TemporaryDirectory;
    using pathvec_t = std::vector<path>;
    
    TEST_CASE("[hdf5-io] Read PNG and JPEG files and write as individual HDF5 binary store files",
              "[hdf5-read-png-jpeg-write-individual-hdf5-files]")
    {
        TemporaryDirectory td("test-hdf5-io");
        path basedir(im::test::basedir);
        const pathvec_t pngs = basedir.list("*.png");
        const pathvec_t jpgs = basedir.list("*.jpg");
        
        std::for_each(pngs.begin(), pngs.end(), [&basedir, &td](path const& p) {
            auto png = im::halide::read(basedir/p);
            path np = td.dirpath/p;
            path npext = np + ".hdf5";
            im::halide::write(png, npext);
        });
        
        std::for_each(jpgs.begin(), jpgs.end(), [&basedir, &td](path const& p) {
            auto jpg = im::halide::read(basedir/p);
            path np = td.dirpath/p;
            path npext = np + ".hdf5";
            im::halide::write(jpg, npext);
        });
        
        const pathvec_t hdfs = td.dirpath.list("*.hdf5");
        CHECK(hdfs.size() == pngs.size() + jpgs.size());
        
        std::for_each(hdfs.begin(), hdfs.end(), [&basedir, &td](path const& p) {
            path np = td.dirpath/p;
            auto hdf = im::halide::read(np);
            REQUIRE(hdf.dim(0) > 0);
            REQUIRE(hdf.dim(1) > 0);
        });
        
    }
    
    TEST_CASE("[hdf5-io] Read TIFF files and write as individual HDF5 binary store files",
              "[hdf5-read-tiff-write-individual-hdf5-files]")
    {
        TemporaryDirectory td("test-hdf5-io");
        path basedir(im::test::basedir);
        const pathvec_t tifs = basedir.list("*.tif*");
        
        std::for_each(tifs.begin(), tifs.end(), [&basedir, &td](path const& p) {
            auto tif = im::halide::read(basedir/p);
            path np = td.dirpath/p;
            path npext = np + ".hdf5";
            im::halide::write(tif, npext);
        });
        
        const pathvec_t hdfs = td.dirpath.list("*.hdf5");
        CHECK(hdfs.size() == tifs.size());
        
        std::for_each(hdfs.begin(), hdfs.end(), [&basedir, &td](path const& p) {
            path np = td.dirpath/p;
            auto hdf = im::halide::read(np);
            REQUIRE(hdf.dim(0) > 0);
            REQUIRE(hdf.dim(1) > 0);
        });
        
    }
  
};