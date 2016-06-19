
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <unordered_map>

#include <libimread/libimread.hpp>
#include <libimread/image.hh>
#include <libimread/imageview.hh>
#include <libimread/histogram.hh>
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
    using im::ImageView;
    using im::Histogram;
    using unique_t = std::unique_ptr<ImageView>;
    using shared_t = std::shared_ptr<ImageView>;
    using histogram_t = std::shared_ptr<Histogram>;
    
    TEST_CASE("[imageview] Create shared ImageView from an Image",
              "[imageview-create-shared-imageview-from-image]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        const std::vector<path> jpgs = basedir.list("*.jpg");
        
        std::for_each(pngs.begin(), pngs.end(), [&basedir](path const& p) {
            auto png = im::halide::unique(basedir/p);
            shared_t png_view = std::make_shared<ImageView>(png.get());
            
            std::unique_ptr<HybridImage> hybrid(new HybridImage(im::halide::read(basedir/p)));
            shared_t unrelated_png_view = std::make_shared<ImageView>(hybrid.get());
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
        
        std::for_each(jpgs.begin(), jpgs.end(), [&basedir](path const& p) {
            auto jpg = im::halide::unique(basedir/p);
            shared_t jpg_view = std::make_shared<ImageView>(jpg.get());
            
            std::unique_ptr<HybridImage> hybrid(new HybridImage(im::halide::read(basedir/p)));
            shared_t unrelated_jpg_view = std::make_shared<ImageView>(hybrid.get());
            shared_t another_jpg_view = jpg_view->shared();
            
            CHECK(jpg_view->nbytes() == unrelated_jpg_view->nbytes());
            CHECK(jpg_view->ndims() == unrelated_jpg_view->ndims());
            CHECK(jpg_view->is_signed() == unrelated_jpg_view->is_signed());
            CHECK(jpg_view->is_floating_point() == unrelated_jpg_view->is_floating_point());
            
            CHECK(jpg_view->nbytes() == another_jpg_view->nbytes());
            CHECK(jpg_view->ndims() == another_jpg_view->ndims());
            CHECK(jpg_view->is_signed() == another_jpg_view->is_signed());
            CHECK(jpg_view->is_floating_point() == another_jpg_view->is_floating_point());
        });
        
    }
    
    
    TEST_CASE("[imageview] Calculate Histogram data from Image using ImageView",
              "[imageview-calculate-histogram-data-from-image-using-imageview]")
    {
        path basedir(im::test::basedir);
        const std::vector<path> pngs = basedir.list("*.png");
        const std::vector<path> jpgs = basedir.list("*.jpg");
        std::unordered_map<path, float> entropies;
        
        std::for_each(pngs.begin(), pngs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            auto png = im::halide::unique(imagepath);
            shared_t png_view = std::make_shared<ImageView>(png.get());
            histogram_t histo = png_view->histogram();
            CHECK(histo->entropy() != 0.00);
            CHECK(!std::isnan(histo->entropy()));
            entropies.insert({ imagepath, histo->entropy() });
        });
        
        std::for_each(jpgs.begin(), jpgs.end(), [&](path const& p) {
            path imagepath = basedir/p;
            auto jpg = im::halide::unique(imagepath);
            shared_t jpg_view = std::make_shared<ImageView>(jpg.get());
            histogram_t histo = jpg_view->histogram();
            CHECK(histo->entropy() != 0.00);
            CHECK(!std::isnan(histo->entropy()));
            entropies.insert({ imagepath, histo->entropy() });
        });
        
        for (auto const& p : entropies) {
            auto image = im::halide::read(p.first.str());
            CHECK(p.second == image.entropy());
        }
        
    }
    
};
