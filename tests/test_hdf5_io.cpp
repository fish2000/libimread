
#include <algorithm>
#include <vector>
#include <unordered_map>

#include <libimread/libimread.hpp>
#include <libimread/halide.hh>
#include <libimread/metadata.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/file.hh>
#include <libimread/filehandle.hh>
#include <libimread/IO/hdf5.hh>

#include "helpers/collect.hh"
#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using filesystem::path;
    using filesystem::TemporaryDirectory;
    using HybridImage = im::HybridImage<uint8_t>;
    using Metadata = im::Metadata;
    using pathvec_t = std::vector<path>;
    
    TEST_CASE("[hdf5-io] Read PNG and JPEG files and write as individual HDF5 binary store files",
              "[hdf5-read-png-jpeg-write-individual-hdf5-files]")
    {
        TemporaryDirectory td("test-hdf5-io");
        path basedir(im::test::basedir);
        const pathvec_t pngs = basedir.list("*.png");
        const pathvec_t jpgs = basedir.list("*.jpg");
        std::unordered_map<path, HybridImage*> images;
        
        std::for_each(pngs.begin(), pngs.end(), [&](path const& p) {
            auto png = im::halide::read(basedir/p);
            path np = td.dirpath/p;
            path npext = np + ".hdf5";
            im::halide::write(png, npext);
            images[npext] = new HybridImage(png);
        });
        
        std::for_each(jpgs.begin(), jpgs.end(), [&](path const& p) {
            auto jpg = im::halide::read(basedir/p);
            path np = td.dirpath/p;
            path npext = np + ".hdf5";
            im::halide::write(jpg, npext);
            images[npext] = new HybridImage(jpg);
        });
        
        const pathvec_t hdfs = td.dirpath.list("*.hdf5");
        CHECK(hdfs.size() == pngs.size() + jpgs.size());
        CHECK(hdfs.size() == images.size());
        
        std::for_each(hdfs.begin(), hdfs.end(), [&](path const& p) {
            path np = td.dirpath/p;
            auto hdf = im::halide::read(np);
            CHECK(hdf.dim(0) == images[np]->dim(0));
            CHECK(hdf.dim(1) == images[np]->dim(1));
            CHECK(hdf.dim(2) == images[np]->dim(2));
            CHECK(hdf.stride(0) == images[np]->stride(0));
            CHECK(hdf.stride(1) == images[np]->stride(1));
            CHECK(hdf.stride(2) == images[np]->stride(2));
            CHECK(hdf.size() == images[np]->size());
            
            // CHECK(std::equal(data.begin(),     data.end(),
            //                  fulldata.begin(), fulldata.end(),
            //                  std::equal_to<byte>()));
            
            CHECK(COLLECT(np));
            
            Metadata* meta = dynamic_cast<Metadata*>(&hdf);
            CHECK(meta != nullptr);
            
            /// comparing as integer:
            CHECK(hdf.dim(0) == std::stoi(meta->get("dim0")));
            CHECK(hdf.dim(1) == std::stoi(meta->get("dim1")));
            CHECK(hdf.dim(2) == std::stoi(meta->get("dim2")));
            CHECK(hdf.stride(0) == std::stoi(meta->get("stride0")));
            CHECK(hdf.stride(1) == std::stoi(meta->get("stride1")));
            CHECK(hdf.stride(2) == std::stoi(meta->get("stride2")));
            
            /// comparing as std::string:
            CHECK(std::to_string(hdf.dim(0)) == meta->get("dim0"));
            CHECK(std::to_string(hdf.dim(1)) == meta->get("dim1"));
            CHECK(std::to_string(hdf.dim(2)) == meta->get("dim2"));
            CHECK(std::to_string(hdf.stride(0)) == meta->get("stride0"));
            CHECK(std::to_string(hdf.stride(1)) == meta->get("stride1"));
            CHECK(std::to_string(hdf.stride(2)) == meta->get("stride2"));
            
            // WTF("Values: ", meta->values.to_string());
            
        });
        
        std::for_each(images.begin(), images.end(), [](auto const& kv) { delete kv.second; });
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