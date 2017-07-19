
#include <map>
#include <utility>
#include <iostream>
#include <libimread/ext/pystring.hh>
#include "libimread-config.hh"
#include "docopt.h"

static const char USAGE[] = R"(Configuration information tool for libimread

    Usage:
      imread-config (--prefix          |
                     --exec_prefix     |
                     --includes        |
                     --libs            |
                     --cflags          |
                     --ldflags)
      imread-config (-h | --help)
      imread-config --version
    
    Options:
      --prefix        Show install prefix e.g. /usr/local.
      --exec_prefix   Show exec-prefix; should be same as the prefix.
      --includes      Show include flags
                        e.g. -I/usr/include -I/usr/local/include.
      --libs          Show library link flags
                        e.g. -limread -lHalide -framework CoreFoundation.
      --cflags        Show all compiler arguments and include flags.
      --ldflags       Show all library link flags and linker options.
      -h --help       Show this help text.
      --version       Show the libimread version.

)";

static const std::string VERSION = "imread-config ";

#define MATCH_CONFIG_FLAG(flagname)                         \
    if (arg.first == "--" # flagname) {                     \
        std::cout << pystring::strip(im::config::flagname)  \
                  << std::endl;                             \
        return EXIT_SUCCESS;                                \
    }

int main(int argc, const char** argv) {
    using value_t = docopt::value;
    using optmap_t = std::map<std::string, value_t>;
    using optpair_t = std::pair<std::string, value_t>;
    value_t truth(true);
    optmap_t args;
    optmap_t raw_args = docopt::docopt(USAGE,
                                      { argv + 1, argv + argc }, true, /// show help
                                       VERSION + im::config::version);
    
    /// filter out all docopt parse artifacts,
    /// leaving only things beginning with "--"
    std::copy_if(raw_args.begin(),
                 raw_args.end(),
                 std::inserter(args, args.begin()),
              [](optpair_t const& arg) { return arg.first.substr(0, 2) == "--"; });
    
    // for (auto const& arg : args) {
    //     std::cout << arg.first << " --> " << arg.second << std::endl;
    // }
    
    /// print the value for the truthy option flag
    for (optpair_t const& arg : args) {
        if (arg.second == truth) {
            
            MATCH_CONFIG_FLAG(prefix);
            MATCH_CONFIG_FLAG(exec_prefix);
            MATCH_CONFIG_FLAG(includes);
            MATCH_CONFIG_FLAG(libs);
            MATCH_CONFIG_FLAG(cflags);
            MATCH_CONFIG_FLAG(ldflags);
            
        }
    }
    
    std::cout << "Error: no recognizable option passed"
              << std::endl
              << "Use -h or --help to see valid options"
              << std::endl;
    return EXIT_FAILURE;
    
}
