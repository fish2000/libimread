
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_set>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/imageformat.hh>
#include <libimread/options.hh>
#include <libimread/symbols.hh>
#include <iod/json.hh>

#include "include/catch.hpp"

namespace {
    
    using im::byte;
    using im::ImageFormat;
    using im::Options;
    using bytevec_t = std::vector<byte>;
    using stringvec_t = std::vector<std::string>;
    using formatset_t = std::unordered_set<ImageFormat>;
    
    TEST_CASE("[imageformat-options] Check registered formats",
              "[imageformat-options-check-registered-formats]")
    {
        auto DMV = ImageFormat::registry();
        formatset_t formats;
        stringvec_t names;
        std::string joined;
        int idx = 0,
            max = 0;
        
        std::transform(DMV.begin(),
                       DMV.end(),
                       std::back_inserter(names),
                    [](auto const& registrant) { return registrant.first; });
        
        joined = std::accumulate(names.begin(),
                                 names.end(),
                                 std::string{},
                        [&names](std::string const& lhs,
                                 std::string const& rhs) {
            return lhs + rhs + (rhs == names.back() ? "" : ", ");
        });
        
        // WTF("",
        //     "REGISTRY:",
        //     FF("\t contains %i formats:", max = names.size()),
        //     FF("\t %s", joined.c_str()));
        
        for (auto it = names.begin();
            it != names.end() && idx < max;
            ++it) { std::string const& format = *it;
                auto format_ptr = ImageFormat::named(format);
                Options opts = format_ptr->get_options();
                
                // WTF("",
                //     FF("FORMAT: %s", format.c_str()),
                //     "As JSON:",
                //     opts.format(), "",
                //     "As encoded IOD:",
                //     iod::json_encode(format_ptr->options),
                //     iod::json_encode(format_ptr->capacity));
                
            ++idx; }
        
        
    }
    
    
    
};
