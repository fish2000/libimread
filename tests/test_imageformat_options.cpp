
#define CATCH_CONFIG_FAST_COMPILE

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
        
        std::string asterisks(160, '*');
        
        for (auto it = names.begin();
            it != names.end() && idx < max;
            ++it) { std::string const& format = *it;
                auto format_ptr = ImageFormat::named(format);
                Options opts = format_ptr->get_options();
                stringvec_t subgroups = opts.subgroups();
                
                // WTF("",
                //     ansi::lightred.str("Format name: " + format),
                //     ansi::red.str("As formatted JSON:"),
                //     FF("\n%s\n%s\n%s", asterisks.c_str(),
                //                        opts.format().c_str(),
                //                        asterisks.c_str()),
                //     FF("\nSubgroups: %s", join(subgroups).c_str()));
                
                // WTF("",
                //     ansi::lightred.str("Format name: " + format),
                //     ansi::red.str("As encoded IOD:"),
                //     FF("OPTIONS  » %s", iod::json_encode(format_ptr->options).c_str()),
                //     FF("CAPACITY » %s", iod::json_encode(format_ptr->capacity).c_str()));
                
                // WTF("SUBGROUPS:", join(subgroups));
                
                // for (std::string const& subgroup : subgroups) {
                //     opts.regroup(subgroup);
                // }
                
                opts.flatten();
                
                // WTF("",
                //     ansi::lightred.str("Format name: " + format),
                //     ansi::red.str("As formatted JSON:"),
                //     FF("\n%s\n%s\n%s", asterisks.c_str(),
                //                        opts.format().c_str(),
                //                        asterisks.c_str()));
                
                opts.extrude();
                
                // WTF("",
                //     ansi::lightred.str("Format name: " + format),
                //     ansi::red.str("As formatted JSON:"),
                //     FF("\n%s\n%s\n%s", asterisks.c_str(),
                //                        opts.format().c_str(),
                //                        asterisks.c_str()));
                
    ++idx; }
    }
    
    
    
};
