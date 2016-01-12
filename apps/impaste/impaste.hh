/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef APPS_IMPASTE_HH_
#define APPS_IMPASTE_HH_

#include <string>
#include <type_traits>
#include <libimread/libimread.hpp>
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#include <libimread/objc-rt/appkit.hh>
#import <libimread/ext/categories/NSString+STL.hh>
#import <libimread/ext/categories/NSURL+IM.hh>
#import <libimread/ext/categories/NSBitmapImageRep+IM.hh>
#import <libimread/ext/categories/NSData+IM.hh>
#include <libimread/ext/filesystem/path.h>

/// printffery
#define IMPASTE_DEBUG 0

/// App delegate only implements `applicationWillTerminate:`
@interface AXAppDelegate : NSObject <NSApplicationDelegate> {}
- (void) applicationWillTerminate:(NSApplication*)application;
@end

/// Base thread class with options
@interface AXThread : NSThread {}
@property (nonatomic, strong) NSDictionary* options;
- (instancetype) initWithOptions:(NSDictionary*)optionsDict;
@end


/// NSThread declarations and definitions,
/// one(ish) for each CLI option

@interface AXCheckThread : AXThread {}
- (void) main;
@end

@interface AXDryRunThread : AXThread {}
- (void) main;
@end

@interface AXImageSaveThread : AXThread {}
- (void) main;
@end

namespace objc {
    
    template <typename OCThreadType> inline
    void run_thread(NSDictionary* options = @{}) {
        @autoreleasepool {
            [NSApplication sharedApplication];
            [[[OCThreadType alloc] initWithOptions:options] start];
            AXAppDelegate* delegate = [[AXAppDelegate alloc] init];
            NSApp.delegate = delegate;
            [NSApp run];
        };
    }
    
}


namespace im {
    
    namespace config {
        
        const std::string version = IM_VERSION;
        
    };
    
};


#endif /// APPS_IMPASTE_HH_