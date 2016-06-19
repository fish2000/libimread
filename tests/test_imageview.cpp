
#include <string>
#include <vector>
// #include <numeric>
#include <algorithm>
#include <memory>

#include <libimread/libimread.hpp>
// #include <libimread/errors.hh>
#include <libimread/image.hh>
#include <libimread/imageview.hh>
// #include <libimread/base.hh>
#include <libimread/halide.hh>
#include <libimread/image.hh>
#include <libimread/ext/filesystem/path.h>
#include <libimread/ext/filesystem/temporary.h>
#include <libimread/file.hh>
#include <libimread/filehandle.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

namespace {
    
    using im::byte;
    using im::Image;
    using filesystem::path;
    using unique_image_t = std::unique_ptr<Image>;
    using shared_image_t = std::shared_ptr<Image>;
    using bytevec_t = std::vector<byte>;
    using stringvec_t = std::vector<std::string>;
    using HybridImage = im::HybridImage<byte>;
    using unique_hybridimage_t = std::unique_ptr<HybridImage>;
    using shared_hybridimage_t = std::shared_ptr<HybridImage>;
    // using ImageView = im::ImageView<HybridImage>;
    using im::ImageView;
    using unique_t = std::unique_ptr<ImageView>;
    using shared_t = std::shared_ptr<ImageView>;
    
    TEST_CASE("[imageview] Create a shared image view from an Image unique_ptr",
              "[imageview-shared-imageview-from-image-unique_ptr]")
    {
        // filesystem::TemporaryDirectory td("test-imageview");
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        const std::vector<path> jpgs = basedir.list("*.jpg");
        
        std::for_each(pngs.begin(), pngs.end(), [&basedir](path const& p) {
            using unique_tag = ImageView::Tag::Unique;
            // unique_image_t png_unique = ;
            shared_t png_view = std::make_shared<ImageView>(std::move(
                                im::halide::unique(basedir/p)), unique_tag{});
            
            // CHECK(png_view->nbits() == png_unique->nbits());
            // auto whatisaid = png_unique->nbits();
            // CHECK(png_view->nbits() == whatisaid);
            // auto whatisaid = png_view->nbits();
            // CHECK(png_unique->nbits() == whatisaid);
            
            HybridImage hybrid = im::halide::read(basedir/p);
            HybridImage* heaprid = new HybridImage(hybrid);
            // ImageView view(heaprid); /// takes ownership
            // shared_t unrelated_png_view = std::make_shared<ImageView>();
            shared_t unrelated_png_view = std::make_shared<ImageView>(heaprid); /// takes ownership
            shared_t another_png_view = png_view->shared();
            
            CHECK(png_view->nbytes() == unrelated_png_view->nbytes());
            CHECK(png_view->ndims() == unrelated_png_view->ndims());
            CHECK(png_view->is_signed() == unrelated_png_view->is_signed());
            CHECK(png_view->is_floating_point() == unrelated_png_view->is_floating_point());
            
            CHECK(png_view->nbytes() == another_png_view->nbytes());
            CHECK(png_view->ndims() == another_png_view->ndims());
            CHECK(png_view->is_signed() == another_png_view->is_signed());
            CHECK(png_view->is_floating_point() == another_png_view->is_floating_point());
        });
        
        // std::for_each(jpgs.begin(), jpgs.end(), [&basedir](path const& p) {
        //     // auto jpg = im::halide::read(basedir/p);
        //     using unique_tag = ImageView::Tag::Unique;
        //     unique_image_t jpg_unique = im::halide::unique(basedir/p);
        //     shared_t jpg_view = std::make_shared<ImageView>(std::move(jpg_unique), unique_tag{});
        //     CHECK(jpg_view->nbits() == jpg_unique->nbits());
        //     CHECK(jpg_view->nbytes() == jpg_unique->nbytes());
        //     CHECK(jpg_view->ndims() == jpg_unique->ndims());
        //     CHECK(jpg_view->is_signed() == jpg_unique->is_signed());
        //     CHECK(jpg_view->is_floating_point() == jpg_unique->is_floating_point());
        // });
        
        // const std::vector<path> hdfs = td.dirpath.list("*.hdf5");
        // CHECK(hdfs.size() == pngs.size() + jpgs.size());
        //
        // std::for_each(hdfs.begin(), hdfs.end(), [&basedir, &td](path const& p) {
        //     path np = td.dirpath/p;
        //     auto hdf = im::halide::read(np);
        //     REQUIRE(hdf.dim(0) > 0);
        //     REQUIRE(hdf.dim(1) > 0);
        // });
        
    }
    
    
    
};
