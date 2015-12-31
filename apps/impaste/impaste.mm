
#include <map>
#include <string>
#include <iostream>

#include "impaste.hh"
#include "docopt.h"

/// NSThread declarations and definitions,
/// one(ish) for each CLI option

@protocol AXCopyReceiver <NSObject>
- (NSArray<NSString*>*) 
@end

NSImage* getit() {
    NSPasteboard* board = [NSPasteboard generalPasteboard];
    // NSArray* classes = @[ [NSImage class] ];
    // NSDictionary* options = @{};
    
    BOOL ok = [board canReadObjectsForClasses:@[ [NSImage class] ]
                                      options:@{}];
    if (!ok) { return nil; }
    
    // NSImage* image = [board readObjectsForClasses:@[ [NSImage class] ]
    //                                       options:@{}][0];
    // return image;
    return [board readObjectsForClasses:@[ [NSImage class] ]
                                options:@{}][0];
}


@interface AXCheckThread : NSThread {}
@end

@interface AXDryRunThread : NSThread {}
@end

@interface AXImageSaveThread : NSThread {}
@end

@implementation AXCheckThread : NSThread
- (void) main {
    
}
@end

@implementation AXDryRunThread : NSThread
- (void) main {
    
}
@end

@implementation AXImageSaveThread : NSThread
- (void) main {
    
}
@end


/// The docopt help string defines the options available:

static const char USAGE[] = R"(Paste image data to imgur.com or to a file

    Usage:
        impaste       [options]
        impaste       (-h | --help)
        impaste        --version
    
    Options:
        -c --check          Check and report on the pasteboard contents.
        -d --dry-run        Don't actually do anything, but pretend.
        -o FILE,
        --output=FILE       Save pasteboard image to a file.
        -h --help           Show this help screen.
        --version           Show version.

)";

const std::string VERSION = "impaste ";

int main(int argc, const char** argv) {
    using value_t = docopt::value;
    using optmap_t = std::map<std::string, value_t>;
    using optpair_t = std::pair<std::string, value_t>;
    value_t truth(true);
    optmap_t args;
    optmap_t raw_args = docopt::docopt(USAGE, { argv + 1, argv + argc },
                                       true, /// show help
                                       VERSION + im::config::version);
    
    std::cout << "RAW ARGS:" << std::endl;
    for (optpair_t const& arg : raw_args) {
        std::cout << arg.first << " --> " << arg.second << std::endl;
    }
    std::cout << std::endl;
    
    /// filter out all docopt parse artifacts,
    /// leaving only things beginning with "--"
    std::copy_if(raw_args.begin(), raw_args.end(),
                 std::inserter(args, args.begin()),
                 [](const optpair_t& p) { return p.first.substr(0, 2) == "--"; });
    
    std::cout << "FILTERED ARGS:" << std::endl;
    for (optpair_t const& arg : args) {
        std::cout << arg.first << " --> " << arg.second << std::endl;
    }
    std::cout << std::endl;
    
    /// print the value for the truthy option flag
    for (optpair_t const& arg : args) {
        if (arg.second == truth) {
            if (arg.first == "--check") {
                /* DO CHECK */
                @autoreleasepool {
                    [NSApplication sharedApplication];
                    [NSApp run];
                }
                break;
            } else if (arg.first == "--dry-run") {
                /* DO DRY RUN */
                @autoreleasepool {
                    [NSApplication sharedApplication];
                    [NSApp run];
                }
                break;
            } else if (arg.first == "--output") {
                /* DO FILE OUTPUT */
                @autoreleasepool {
                    [NSApplication sharedApplication];
                    [NSApp run];
                }
                break;
            }
        }
    }

}
