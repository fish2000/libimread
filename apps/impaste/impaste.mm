
#include <cstdlib>

#include <map>
#include <atomic>
#include <utility>
#include <iostream>
#include <algorithm>

#include "impaste.hh"
#include "docopt.h"

/// return value as static global (ugh I know I know)
static std::atomic<int> return_value(EXIT_SUCCESS);

/// App delegate

@implementation AXAppDelegate
- (void) applicationWillTerminate:(NSApplication *)application {
    
    std::cout << "Exiting with status: "
              << return_value.load()
              << std::endl;
    
}
@end

/// NSThread declarations and definitions,
/// one(ish) for each CLI option

@implementation AXCheckThread : NSThread
- (void) main {
    
    std::cout << "Checking default NSPasteboard for images ..."
              << std::endl;
    
    BOOL ok = objc::appkit::can_paste<NSImage>();
    
    if (ok) {
        
        std::cout << "Pasteboard contains useable image data [go nuts!]"
                  << std::endl;
        return_value.store(EXIT_SUCCESS);
        
    } else { /// THIS IS NOT TOWARD
        
        std::cout << "No useable image data found [sorry dogg]"
                  << std::endl;
        return_value.store(EXIT_FAILURE);
        
    }
    
    /// exit from thread
    [NSApp terminate:self];
    return;
    
}
@end

@implementation AXDryRunThread : NSThread
- (void) main {
    
    std::cout << "Dry run not implemented yet, exiting"
              << std::endl;
    return_value.store(EXIT_FAILURE);
    
    /// exit from thread
    [NSApp terminate:self];
    return;
    
}
@end

@implementation AXImageSaveThread : NSThread
- (void) main {
    
    std::cout << "Image save not implemented yet, exiting"
              << std::endl;
    return_value.store(EXIT_FAILURE);
    
    /// exit from thread
    [NSApp terminate:self];
    return;
    
}
@end


/// The docopt help string defines the options available:

static const char USAGE[] = R"(Paste image data to imgur.com or to a file

    Usage:
        impaste       (-c      | --check)           [options]
        impaste       (-d      | --dry-run)         [options]
        impaste       (-o FILE | --output=FILE)     [options]
        impaste       (-h      | --help)
        impaste                  --version
    
    Options:
        -c --check          Check and report on the pasteboard contents.
        -d --dry-run        Don't actually do anything, but pretend.
        -o FILE,
        --output=FILE       Save pasteboard image to a file.
        -v --verbose        Print more information.
        -h --help           Show this help screen.
        --version           Show version.

)";

const std::string VERSION = "impaste ";


int main(int argc, const char** argv) {
    using value_t = docopt::value;
    using optmap_t = std::map<std::string, value_t>;
    using optpair_t = std::pair<std::string, value_t>;
    value_t truth(1);
    optmap_t args;
    optmap_t raw_args = docopt::docopt(USAGE, { argv + 1, argv + argc },
                                       true, /// show help
                                       VERSION + im::config::version);
    
    #if IMPASTE_DEBUG
        std::cout << "RAW ARGS:" << std::endl;
        for (optpair_t const& arg : raw_args) {
            std::cout << arg.first << " --> " << arg.second << std::endl;
        }
        std::cout << std::endl;
    #endif
    
    /// filter out all docopt parse artifacts,
    /// leaving only things beginning with "--"
    std::copy_if(raw_args.begin(), raw_args.end(),
                 std::inserter(args, args.begin()),
                 [](optpair_t const& p) { return p.first.substr(0, 2) == "--"; });
    
    #if IMPASTE_DEBUG
        std::cout << "FILTERED ARGS:" << std::endl;
        for (optpair_t const& arg : args) {
            std::cout << arg.first << " --> " << arg.second << std::endl;
        }
        std::cout << std::endl;
    #endif
    
    /// print the value for the truthy option flag
    for (optpair_t const& arg : args) {
        if (arg.second == truth) {
            if (arg.first == "--check") {
                /* DO CHECK */
                objc::run_thread<AXCheckThread>();
                break;
            } else if (arg.first == "--dry-run") {
                /* DO DRY RUN */
                objc::run_thread<AXDryRunThread>();
                break;
            } else if (arg.first == "--output") {
                /* DO FILE OUTPUT */
                objc::run_thread<AXImageSaveThread>();
                break;
            }
        }
    }
    
    /// doesn't get called from threads
    std::exit(return_value.load());

}
