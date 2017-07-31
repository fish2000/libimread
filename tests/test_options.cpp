
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <functional>
#include <tuple>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/store.hh>
#include <libimread/options.hh>

#include "include/test_data.hpp"
#include "include/catch.hpp"

extern "C" {
   #include <tiffio.h>
}

namespace {
    
    using im::byte;
    using im::Options;
    using im::OptionsList;
    using bytevec_t = std::vector<byte>;
    using stringvec_t = std::vector<std::string>;
    
    TEST_CASE("[options-container] Check store::is_stringmapper_v values for im::Options instances",
              "[options-container-check-store-is_stringmapper_v-values-im-options-instances]")
    {
        Options opts = {
            {   "yo", "dogg"           },
            {    "i", "heard"          },
            {  "you", "like"           },
            { "list", "initialization" }
        };
        
        REQUIRE(store::is_stringmapper_v<Options>);
        REQUIRE(store::is_stringmapper_v<decltype(opts)>);
        REQUIRE(opts.can_store());
    }
    
    TEST_CASE("[options-container] Check im::Options::count() and im::Options::get() methods on list-constructed instances",
              "[options-container-check-options-count-get-methods-list-constructed-instances]")
    {
        Options opts = {
            {   "yo", "dogg"           },
            {    "i", "heard"          },
            {  "you", "like"           },
            { "list", "initialization" }
        };
        
        CHECK(opts.count() == 4);
        CHECK(opts.get("yo") == "dogg");
        CHECK(opts.get("i") == "heard");
        CHECK(opts.get("you") == "like");
        CHECK(opts.get("list") == "initialization");
        CHECK(opts.get("WAT") == std::string{ NULL_STR });
        CHECK(opts.count() == 4);
        
        {
            store::stringmap optcopy(opts);
            
            CHECK(optcopy.count() == 4);
            CHECK(optcopy.get("yo") == "dogg");
            CHECK(optcopy.get("i") == "heard");
            CHECK(optcopy.get("you") == "like");
            CHECK(optcopy.get("list") == "initialization");
            CHECK(optcopy.get("WAT") == std::string{ NULL_STR });
            CHECK(optcopy.count() == 4);
        }
        
        {
            store::stringmap optfix(opts, "prefix");
            
            CHECK(optfix.count() == 4);
            CHECK(optfix.get("prefix:yo") == "dogg");
            CHECK(optfix.get("prefix:i") == "heard");
            CHECK(optfix.get("prefix:you") == "like");
            CHECK(optfix.get("prefix:list") == "initialization");
            CHECK(optfix.get("prefix:WAT") == std::string{ NULL_STR });
            CHECK(optfix.count() == 4);
        }
        
    }
    
    TEST_CASE("[options-container] Calculate a prefix histogram from an im::Options instance",
              "[options-container-calculate-prefix-histogram-options-instance]")
    {
        std::string large_data(4096, '*');
        std::function<char const*(bool)> booleanizer = [](bool value) {
            return value ? "true" : "false";
        };
        
        Options typical = {
            {  "jpg:quality",               std::to_string(65)              }, /// jpg = 1  *
            {  "png:compression_level",     std::to_string(9)               }, /// png = 1
            {  "tiff:compress",             booleanizer(true)               }, /// tiff = 1
            {  "tiff:horizontal-predictor", booleanizer(false)              }, /// tiff = 2
            {  "metadata",                  std::string(large_data)         }, /// <none>
            {  "tiff:XResolution",          std::to_string(72)              }, /// tiff = 3
            {  "tiff:YResolution",          std::to_string(72)              }, /// tiff = 4
            {  "tiff:XResolutionUnit",      std::to_string(RESUNIT_INCH)    }, /// tiff = 5 *
            {  "png:ios-premultiply",       booleanizer(true)               }, /// png = 2  *
            {  "md:icc",                    std::string(large_data)         }, /// md = 1
            {  "md:xmp-data",               std::string(large_data)         }, /// md = 2
            {  "md:xmp-sidecar",            booleanizer(true)               }, /// md = 3
            {  "md:exif",                   std::string(large_data)         }, /// md = 4
            {  "md:thumbnail",              std::string(large_data)         }  /// md = 5   *
        };                                                                     /// TOTAL = 14
        
        CHECK(typical.prefixcount("md") == 5);
        CHECK(typical.prefixcount("tiff") == 5);
        CHECK(typical.prefixcount("png") == 2);
        CHECK(typical.prefixcount("jpg") == 1);
        CHECK(typical.prefixcount("WAT") == 0);
        
        /// special overload,
        /// for counting keys without prefixes:
        CHECK(typical.prefixcount() == 1);
        
        {
            auto ratios_tuple = typical.ratios();
            
            // WTF("RATIOS:",
            //     FF("\n\tunprefixed: %f\n\tprefixed: %f\n\ttotal: %i", std::get<0>(ratios_tuple),
            //                                                           std::get<1>(ratios_tuple),
            //                                                           std::get<2>(ratios_tuple)),
            // FF("\n\tunprefixed: %i%%\n\tprefixed: %i%%\n\ttotal: %i", static_cast<int>(std::get<0>(ratios_tuple) * 100),
            //                                                           static_cast<int>(std::get<1>(ratios_tuple) * 100),
            //                                                           std::get<2>(ratios_tuple))
            //                                                           );
            
            CHECK(std::get<2>(ratios_tuple) == 14);
            CHECK(std::get<2>(ratios_tuple) == typical.count());
            
            /// These next two int-percentage results were figured out by uncommentding the above
            /// block call to WTF(…) and observing the results.
            CHECK(static_cast<int>(std::get<0>(ratios_tuple) * 100) == 7);   ///  7% unprefixed
            CHECK(static_cast<int>(std::get<1>(ratios_tuple) * 100) == 92);  /// 92% prefixed
        }
        
        prefixgram_t prefixgram = typical.prefixgram();
        
        CHECK(prefixgram["md"] == 5);
        CHECK(prefixgram["tiff"] == 5);
        CHECK(prefixgram["png"] == 2);
        CHECK(prefixgram["jpg"] == 1);
        CHECK(prefixgram["WAT"] == 0);
    }
    
} /* namespace (anon.) */
