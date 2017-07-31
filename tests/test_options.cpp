
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
    using ::prefixgram_t;
    using ::ratios_t;
    
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
        
        SECTION("[options-container] » Confirm count()/get() after initializing im::Options with two im::OptionsList temporaries")
        {
            Options optcopy(opts.keylist(),
                            opts.valuelist());
            
            CHECK(optcopy.count() == 4);
            CHECK(optcopy.get("yo") == "dogg");
            CHECK(optcopy.get("i") == "heard");
            CHECK(optcopy.get("you") == "like");
            CHECK(optcopy.get("list") == "initialization");
            CHECK(optcopy.get("WAT") == std::string{ NULL_STR });
            CHECK(optcopy.count() == 4);
        }
        
        SECTION("[options-container] » Confirm count()/get() after initializing im::Options with two im::OptionsList literals")
        {
            OptionsList keylist = { "yo",   "i",     "you",  "list"             };
            OptionsList vallist = { "dogg", "heard", "like", "initialization"   };
            
            /// some rather pointless OptionsList checks,
            /// while we are in the position to do so:
            CHECK(!keylist.can_store());
            CHECK(!vallist.can_store());
            CHECK(keylist.count() == 4);
            CHECK(vallist.count() == 4);
            CHECK(keylist.count() == opts.count());
            CHECK(vallist.count() == opts.count());
            
            Options optcopy(std::move(keylist),
                            std::move(vallist));
            
            CHECK(optcopy.count() == 4);
            CHECK(optcopy.get("yo") == "dogg");
            CHECK(optcopy.get("i") == "heard");
            CHECK(optcopy.get("you") == "like");
            CHECK(optcopy.get("list") == "initialization");
            CHECK(optcopy.get("WAT") == std::string{ NULL_STR });
            CHECK(optcopy.count() == 4);
        }
        
        SECTION("[options-container] » Confirm count()/get() after copying im::Options to a store::stringmap instance, "
                                      "internally using store::value_copy<…>()")
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
        
        SECTION("[options-container] » Confirm count()/get() after copying im::Options to a store::stringmap instance, "
                                      "internally using store::prefix_copy<…>()")
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
            {  "metadata",                  std::string(large_data)         }, /// <none> 1 *
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
        
        SECTION("[options-container] » Confirm total values by examining a `ratios_t` tuple, "
                                      "using im::Options::ratios()")
        {
            ratios_t ratios_tuple = typical.ratios();
            
            // WTF("RATIOS:",
            //     FF("\n\tunprefixed: %i\n\tprefixed: %i\n\ttotal: %i", std::get<2>(ratios_tuple),
            //                                                           std::get<3>(ratios_tuple),
            //                                                           std::get<4>(ratios_tuple)),
            //     FF("\n\tunprefixed: %f\n\tprefixed: %f\n\ttotal: %i", std::get<0>(ratios_tuple),
            //                                                           std::get<1>(ratios_tuple),
            //                                                           std::get<4>(ratios_tuple)),
            // FF("\n\tunprefixed: %i%%\n\tprefixed: %i%%\n\ttotal: %i", static_cast<int>(std::get<0>(ratios_tuple) * 100),
            //                                                           static_cast<int>(std::get<1>(ratios_tuple) * 100),
            //                                                           std::get<4>(ratios_tuple))
            // ); /// end of WTF
            
            CHECK(std::get<4>(ratios_tuple) == 14);
            CHECK(std::get<4>(ratios_tuple) == typical.count());
            
            CHECK(std::get<2>(ratios_tuple) == 1);
            CHECK(std::get<2>(ratios_tuple) == typical.prefixcount());
            CHECK(std::get<3>(ratios_tuple) == 13);
            CHECK(std::get<3>(ratios_tuple) == (typical.count() - typical.prefixcount()));
            
            /// These next two int-percentage results were figured out by un-commenting-out the above
            /// block call to WTF(…) and observing the printed results.
            CHECK(static_cast<int>(std::get<0>(ratios_tuple) * 100) == 7);   ///  7% unprefixed
            CHECK(static_cast<int>(std::get<1>(ratios_tuple) * 100) == 92);  /// 92% prefixed
        }
        
        SECTION("[options-container] » Confirm per-prefix counts by examining a `prefixgram_t` hash map, "
                                      "using im::Options::prefixgram()")
        {
            prefixgram_t prefixgram = typical.prefixgram();
            
            CHECK(prefixgram["md"] == 5);
            CHECK(prefixgram["tiff"] == 5);
            CHECK(prefixgram["png"] == 2);
            CHECK(prefixgram["jpg"] == 1);
            CHECK(prefixgram["WAT"] == 0);
            
            CHECK(prefixgram["md"]   == typical.prefixcount("md"));
            CHECK(prefixgram["tiff"] == typical.prefixcount("tiff"));
            CHECK(prefixgram["png"]  == typical.prefixcount("png"));
            CHECK(prefixgram["jpg"]  == typical.prefixcount("jpg"));
            CHECK(prefixgram["WAT"]  == typical.prefixcount("WAT"));
        }
    }
    
} /* namespace (anon.) */
