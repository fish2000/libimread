
#include <regex>
#include <vector>
#include <memory>

#include <libimread/libimread.hpp>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>
#include <libimread/imagelist.hh>
#include <libimread/formats.hh>
#include <libimread/IO/png.hh>
#include <libimread/IO/gif.hh>
#include <libimread/halide.hh>

#include "helpers/collect.hh"
#include "include/test_data.hpp"
#include "include/catch.hpp"

// std::unique_ptr<ImageFormat> format_ptr = ImageFormat::named("png");
// std::unique_ptr<formats::PNG> png_format = std::make_unique<formats::PNG>();
// std::unique_ptr<formats::GIF> gif_format = std::make_unique<formats::GIF>();

namespace formats = im::format;

namespace {
    
    using filesystem::NamedTemporaryFile;
    using filesystem::TemporaryDirectory;
    using filesystem::path;
    
    using Options = im::options_map;
    using HybridImage = im::HybridImage<uint8_t>;
    using HalideFactory = im::HalideFactory<uint8_t>;
    
    using im::Image;
    using im::ImageList;
    using im::ImageFormat;
    using im::FileSource;
    using im::FileSink;
    
    using pathvec_t = std::vector<path>;
    
    TEST_CASE("[gif-write] Read PNG files and write as individual GIF files",
              "[gif-write-individual-files]")
    {
        TemporaryDirectory td("write-individual-gifs");
        path basedir(im::test::basedir);
        const pathvec_t outglob = basedir.list("output_*.png");
        std::unique_ptr<HalideFactory> factory{ new HalideFactory };
        std::unique_ptr<formats::PNG> png_format{ new formats::PNG };
        std::unique_ptr<formats::GIF> gif_format{ new formats::GIF };
        Options read_options;
        Options write_options;
        
        std::for_each(outglob.begin(), outglob.end(),
                  [&](path const& p) {
            path fullpath = basedir/p;
            path newpath(td.dirpath/p + ".gif");
            path redopath(td.dirpath/p + "-2.gif");
            
            std::unique_ptr<FileSource> source = std::make_unique<FileSource>(fullpath);
            std::unique_ptr<Image> input = png_format->read(source.get(),
                                                            factory.get(),
                                                            png_format->add_options(read_options));
            
            HybridImage first_hybrid(dynamic_cast<HybridImage&>(*input.get()));
            
            // WTF("*** Loaded First HybridImage:",
            //  FF("\t<%i> [w:%i h:%i p:%i]", &first_hybrid,
            //                                 first_hybrid.dim(0),
            //                                 first_hybrid.dim(1),
            //                                 first_hybrid.dim(2)));
            
            {
                std::unique_ptr<FileSink> rewrite = std::make_unique<FileSink>(newpath);
                gif_format->write(dynamic_cast<Image&>(first_hybrid),
                                  rewrite.get(),
                                  gif_format->add_options(write_options));
                
                /// Stash some metadata w/r/t the image source using xattrs:
                rewrite->xattr("im:original_format", "png");
                rewrite->xattr("im:original_path",   fullpath.str());
                rewrite->xattr("im:original_size",   std::to_string(source->size()));
            }
            
            REQUIRE(newpath.is_file());
            // CHECK(COLLECT(newpath));
            
            std::unique_ptr<FileSource> readback = std::make_unique<FileSource>(newpath);
            std::unique_ptr<Image> output = gif_format->read(source.get(),
                                                             factory.get(),
                                                             gif_format->add_options(read_options));
            
            HybridImage second_hybrid(dynamic_cast<HybridImage&>(*output.get()));
            
            // WTF("*** Loaded Second HybridImage:",
            //  FF("\t<%i> [w:%i h:%i p:%i]", &second_hybrid,
            //                                 second_hybrid.dim(0),
            //                                 second_hybrid.dim(1),
            //                                 second_hybrid.dim(2)));
            
            /// Compare first-pass and second-pass image-data properties:
            CHECK(first_hybrid.nbits()                  == second_hybrid.nbits());
            CHECK(first_hybrid.nbytes()                 == second_hybrid.nbytes());
            CHECK(first_hybrid.ndims()                  == second_hybrid.ndims());
            CHECK(first_hybrid.is_signed()              == second_hybrid.is_signed());
            CHECK(first_hybrid.is_floating_point()      == second_hybrid.is_floating_point());
            CHECK(first_hybrid.rowp_stride()            == second_hybrid.rowp_stride());
            CHECK(first_hybrid.size()                   == second_hybrid.size());
            
            /// N.B. these fail to compile because of overload-resolution errors on the methods:
            // CHECK(first_hybrid.width()                  == second_hybrid.width());
            // CHECK(first_hybrid.height()                 == second_hybrid.height());
            
            /// Verify stashed xattr metadata:
            CHECK(newpath.xattr("im:original_format")   == "png");
            CHECK(newpath.xattr("im:original_path")     == fullpath.str());
            CHECK(newpath.xattr("im:original_size")     == std::to_string(source->size()));
            
            /// Compare image content
            // CHECK(first_hybrid.allplanes<byte>()        == second_hybrid.allplanes<byte>());
            // CHECK(first_hybrid.plane<byte>(0)           == second_hybrid.plane<byte>(0));
            
            {
                std::unique_ptr<FileSink> rerewrite = std::make_unique<FileSink>(redopath);
                gif_format->write(dynamic_cast<Image&>(second_hybrid),
                                  rerewrite.get(),
                                  gif_format->add_options(write_options));
                
                /// Stash some metadata w/r/t the image source using xattrs:
                rerewrite->xattr("im:original_format", "gif");
                rerewrite->xattr("im:original_path",   redopath.str());
                rerewrite->xattr("im:original_size",   std::to_string(readback->size()));
            }
            
            REQUIRE(redopath.is_file());
            // CHECK(COLLECT(redopath));
            
            /// Verify stashed xattr metadata:
            CHECK(redopath.xattr("im:original_format")   == "gif");
            CHECK(redopath.xattr("im:original_path")     == redopath.str());
            CHECK(redopath.xattr("im:original_size")     == std::to_string(readback->size()));
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
            HybridImage* halim = new HybridImage(im::halide::read((basedir/p).str()));
            outlist.push_back(halim);
        });
        
        im::halide::write_multi(outlist, composite.str());
        CHECK(composite.filepath.is_file());
    }
    
}