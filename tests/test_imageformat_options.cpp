
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <unordered_set>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/ansicolor.hh>
#include <libimread/imageformat.hh>
#include <libimread/options.hh>
#include <libimread/symbols.hh>
#include <iod/json.hh>

#include <libimread/IO/all.hh>

#include "include/catch.hpp"

namespace {
    
    using im::byte;
    using im::ImageFormat;
    using im::Options;
    using bytevec_t = std::vector<byte>;
    using stringvec_t = std::vector<std::string>;
    using formatset_t = std::unordered_set<ImageFormat>;
    using formatptrs_t = std::unordered_set<std::unique_ptr<ImageFormat>>;
    
    TEST_CASE("[imageformat-options] Check im::ImageFormat unique-pointer hashability",
              "[imageformat-options-check-imageformat-unique_ptr-hashability]")
    {
        formatptrs_t formats;
        
        formats.insert(std::make_unique<im::BMPFormat>());
        formats.insert(std::make_unique<im::GIFFormat>());
        formats.insert(std::make_unique<im::HDF5Format>());
        formats.insert(std::make_unique<im::JPEGFormat>());
        formats.insert(std::make_unique<im::LSMFormat>());
        formats.insert(std::make_unique<im::PNGFormat>());
        formats.insert(std::make_unique<im::PPMFormat>());
        formats.insert(std::make_unique<im::PVRTCFormat>());
        formats.insert(std::make_unique<im::TIFFFormat>());
        formats.insert(std::make_unique<im::WebPFormat>());
        
        CHECK(formats.size() == 10);
        
    }
    
    TEST_CASE("[imageformat-options] Check im::ImageFormat hashability",
              "[imageformat-options-check-imageformat-hashability]")
    {
        formatset_t rvalues, lvalues;
        
        im::BMPFormat   bmp;
        im::GIFFormat   gif;
        im::HDF5Format  hdf5;
        im::JPEGFormat  jpg;
        im::LSMFormat   lsm;
        im::PNGFormat   png;
        im::PPMFormat   ppm;
        im::PVRTCFormat pvr;
        im::TIFFFormat  tiff;
        im::WebPFormat  webp;
        
        lvalues.insert(bmp);
        lvalues.insert(gif);
        lvalues.insert(hdf5);
        lvalues.insert(jpg);
        lvalues.insert(lsm);
        lvalues.insert(png);
        lvalues.insert(ppm);
        lvalues.insert(pvr);
        lvalues.insert(tiff);
        lvalues.insert(webp);
        
        rvalues.insert(std::move(bmp));
        rvalues.insert(std::move(gif));
        rvalues.insert(std::move(hdf5));
        rvalues.insert(std::move(jpg));
        rvalues.insert(std::move(lsm));
        rvalues.insert(std::move(png));
        rvalues.insert(std::move(ppm));
        rvalues.insert(std::move(pvr));
        rvalues.insert(std::move(tiff));
        rvalues.insert(std::move(webp));
        
        CHECK(lvalues.size() == 10);
        CHECK(rvalues.size() == 10);
        
        // CHECK(lvalues != rvalues);
        
        // WTF("LOAD FACTORS:",
        //     FF("lvalues: %f", lvalues.load_factor()),
        //     FF("rvalues: %f", rvalues.load_factor()));
        
        // WTF("BUCKETS:",
        //     FF("lvalues: %u (max %u)", lvalues.bucket_count(), lvalues.max_bucket_count()),
        //     FF("rvalues: %u (max %u)", rvalues.bucket_count(), rvalues.max_bucket_count()));
        
        {
            im::BMPFormat   nbmp;
            im::GIFFormat   ngif;
            im::HDF5Format  nhdf5;
            im::JPEGFormat  njpg;
            im::LSMFormat   nlsm;
            im::PNGFormat   npng;
            im::PPMFormat   nppm;
            im::PVRTCFormat npvr;
            im::TIFFFormat  ntiff;
            im::WebPFormat  nwebp;
            
            auto lhasher = lvalues.hash_function();
            auto rhasher = rvalues.hash_function();
            
            // WTF("HASHES:",
            //     FF("  BMPFormat: %0u, %0u, %0u",   nbmp.hash(), lhasher(nbmp),  rhasher(bmp)),
            //     FF("  GIFFormat: %0u, %0u, %0u",   ngif.hash(), lhasher(ngif),  rhasher(gif)),
            //     FF(" HDF5Format: %0u, %0u, %0u",  nhdf5.hash(), lhasher(nhdf5), rhasher(hdf5)),
            //     FF(" JPEGFormat: %0u, %0u, %0u",   njpg.hash(), lhasher(njpg),  rhasher(jpg)),
            //     FF("  LSMFormat: %0u, %0u, %0u",   nlsm.hash(), lhasher(nlsm),  rhasher(lsm)),
            //     FF("  PNGFormat: %0u, %0u, %0u",   npng.hash(), lhasher(npng),  rhasher(png)),
            //     FF("  PPMFormat: %0u, %0u, %0u",   nppm.hash(), lhasher(nppm),  rhasher(ppm)),
            //     FF("PVRTCFormat: %0u, %0u, %0u",   npvr.hash(), lhasher(npvr),  rhasher(pvr)),
            //     FF(" TIFFFormat: %0u, %0u, %0u",  ntiff.hash(), lhasher(ntiff), rhasher(tiff)),
            //     FF(" WebPFormat: %0u, %0u, %0u",  nwebp.hash(), lhasher(nwebp), rhasher(webp)));
            
            std::hash<std::string> hasher;
            
            // WTF("STRING HASHES:",
            //     FF(" empty string: %u", hasher("")),
            //     FF("“ImageFormat”: %u", hasher("ImageFormat")));
        
        }
        
    }
    
    static std::string join(stringvec_t const& strings) {
        return std::accumulate(strings.begin(),
                               strings.end(),
                               std::string{},
                    [&strings](std::string const& lhs,
                               std::string const& rhs) {
            return lhs + rhs + (rhs.c_str() == strings.back().c_str() ? "" : ", ");
        });
    }
    
    TEST_CASE("[imageformat-options] Check registered formats",
              "[imageformat-options-check-registered-formats]")
    {
        auto DMV = ImageFormat::registry();
        stringvec_t names;
        std::string joined;
        int idx = 0,
            max = 0;
        
        std::transform(DMV.begin(),
                       DMV.end(),
                       std::back_inserter(names),
                    [](auto const& registrant) { return registrant.first; });
        
        joined = join(names);
        
        WTF("",
            "REGISTRY:",
            FF("\t contains %i formats:", max = names.size()),
            FF("\t %s", joined.c_str()));
        
        std::string asterisks(80, '*');
        
        for (auto it = names.begin();
            it != names.end() && idx < max;
            ++it) { std::string const& format = *it;
                auto format_ptr = ImageFormat::named(format);
                Options opts = format_ptr->get_options();
                stringvec_t subgroups = opts.subgroups();
                
                WTF("",
                    ansi::lightred.str("Format name: " + format),
                    ansi::red.str("As formatted JSON:"),
                    FF("\n%s\n%s\n%s", asterisks.c_str(),
                                       opts.format().c_str(),
                                       asterisks.c_str()),
                    FF("\nSubgroups: %s", join(subgroups).c_str()));
                
                // WTF("",
                //     ansi::lightred.str("Format name: " + format),
                //     ansi::red.str("As encoded IOD:"),
                //     FF("OPTIONS  » %s", iod::json_encode(format_ptr->options).c_str()),
                //     FF("CAPACITY » %s", iod::json_encode(format_ptr->capacity).c_str()));
                
                
                
                WTF("SUBGROUPS:", join(subgroups));
    ++idx; }
    }
    
    
    
};
