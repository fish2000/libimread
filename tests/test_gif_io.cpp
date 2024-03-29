
#define CATCH_CONFIG_FAST_COMPILE

#include <regex>
#include <vector>
#include <memory>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
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

namespace formats = im::format;

namespace {
    
    using filesystem::TemporaryName;
    using filesystem::TemporaryDirectory;
    using filesystem::path;
    
    using Options = im::Options;
    using HybridImage = im::HybridImage<uint8_t>;
    using HalideFactory = im::HalideFactory<uint8_t>;
    
    using im::Image;
    using im::ImageList;
    using im::ImageFormat;
    using im::FileSource;
    using im::FileSink;
    
    using pathvec_t = std::vector<path>;
    
    TEST_CASE("[gif-io] Read PNG files and write as individual GIF files",
              "[gif-io-write-individual-files]")
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
    
    // TEST_CASE("[gif-io] WHAT THE HELL PEOPLE",
    //           "[gif-io-what-the-hell-people]") {
    //     ImageList what_the_hell_people, dogg_why_you_even_got_to_do_a_thing;
    //     path basedir(im::test::basedir);
    //     const pathvec_t sequence = basedir.list(std::regex("output_([0-9]+).png"));
    //     std::for_each(sequence.begin(), sequence.end(),
    //               [&](path const& p) {
    //         HybridImage* halim = new HybridImage(im::halide::read((basedir/p).str()));
    //         what_the_hell_people.push_back(halim);
    //
    //         /// confirm lack of pre-computed image sizes:
    //         CHECK(what_the_hell_people.width() == -1);
    //         CHECK(what_the_hell_people.height() == -1);
    //         CHECK(what_the_hell_people.planes() == -1);
    //
    //         try {
    //             TemporaryName composite(".gif");
    //
    //             /// GIFFormat::write_multi(ilist) calls ilist.compute_sizes():
    //             im::halide::write_multi(what_the_hell_people, composite.str());
    //             REQUIRE(composite.pathname.is_file());
    //         } catch (std::bad_alloc& exc) {
    //             WTF("BAD ALLOCATION:",
    //                 FF("\t%s\n", exc.what()));
    //         }
    //
    //     });
    // }
    
    TEST_CASE("[gif-io] Read PNG files and write as a single animated GIF file",
              "[gif-io-multi-animated]")
    {
        TemporaryName composite(".gif");
        std::unique_ptr<HalideFactory> factory{ new HalideFactory };
        std::unique_ptr<formats::GIF> gif_format{ new formats::GIF };
        Options read_options;
        Options write_options;
        ImageList outlist, inlist;
        path basedir(im::test::basedir);
        const pathvec_t sequence = basedir.list(std::regex("output_([0-9]+).png"));
        
        outlist.reserve(sequence.size());
        std::for_each(sequence.begin(),
                      sequence.end(),
                  [&](path const& p) { outlist.push_back(im::halide::unique((basedir/p).str())); });
        
        /// confirm that the image list is the right size:
        CHECK(outlist.size() == sequence.size());
        
        /// confirm lack of pre-computed image sizes:
        CHECK(outlist.width() == -1);
        CHECK(outlist.height() == -1);
        CHECK(outlist.planes() == -1);
        
        outlist.compute_sizes();
        
        /// confirm values for newly pre-computed image sizes:
        CHECK(outlist.width() == outlist[0]->width());
        CHECK(outlist.height() == outlist[0]->height());
        CHECK(outlist.planes() == outlist[0]->planes());
        
        /// GIFFormat::write_multi(ilist) calls ilist.compute_sizes():
        im::halide::write_multi(outlist, composite.str(), write_options);
        REQUIRE(composite.pathname.is_file());
        
        {
            std::unique_ptr<FileSource> source = std::make_unique<FileSource>(composite.pathname);
            inlist = gif_format->read_multi(source.get(),
                                            factory.get(),
                                            gif_format->add_options(read_options));
        }
        
        /// confirm the frame counts match:
        REQUIRE(inlist.size() == sequence.size());
        REQUIRE(inlist.size() == outlist.size());
        
        /// check computed image sizes:
        CHECK(inlist.width() == outlist.width());
        CHECK(inlist.height() == outlist.height());
        CHECK(inlist.planes() == outlist.planes());
        
    }
    
}