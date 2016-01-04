/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef APPS_IMPASTE_HH_
#define APPS_IMPASTE_HH_

#include <string>
#include <type_traits>
#include <exception>
#include <stdexcept>
#include <libimread/libimread.hpp>
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <libimread/ext/categories/NSString+STL.hh>
#import <libimread/ext/categories/NSURL+IM.hh>

/// printffery
#define IMPASTE_DEBUG 0

/// App delegate only implements `applicationWillTerminate:`
@interface AXAppDelegate : NSObject <NSApplicationDelegate> {}
- (void) applicationWillTerminate:(NSApplication *)application;
@end

/// Base thread class with options
@interface AXThread : NSThread {}
@property (nonatomic, strong) NSDictionary* options;
- (instancetype) initWithOptions:(NSDictionary *)optionsDict;
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
    
    namespace appkit {
        
        template <typename OCType>
        BOOL can_paste(NSPasteboard* board = nil) noexcept {
            if (!board) { board = [NSPasteboard generalPasteboard]; }
            return [board canReadObjectForClasses:@[ [OCType class] ]
                                          options:@{}];
        }
        
        template <typename OCType>
        __attribute__((ns_returns_retained))
        typename std::add_pointer_t<OCType> paste(NSPasteboard* board = nil) noexcept {
            if (!board) { board = [NSPasteboard generalPasteboard]; }
            if (!can_paste<OCType>(board)) { return nil; }
            NSArray* out = [board readObjectsForClasses:@[ [OCType class] ]
                                                options:@{}];
            
            /// array is nil on error -- but empty if the call to
            /// `readObjectsForClasses:options:` comes up short… SOOO:
            return out == nil ? nil : out[0];
        }
        
        template <typename ...OCTypes>
        BOOL copy_to(NSPasteboard* board, OCTypes... objects) noexcept {
            if (!board) { board = [NSPasteboard generalPasteboard]; }
            __attribute__((__unused__))
            NSInteger changecount = [board clearContents];
            return [board writeObjects:@[ objects... ]];
        }
        
        template <typename ...OCTypes> inline
        BOOL copy(OCTypes... objects) noexcept {
            return copy_to<OCTypes...>([NSPasteboard generalPasteboard],
                                        objects...);
        }
        
    }
    
}


namespace im {
    
    namespace config {
        
        const std::string version = IM_VERSION;
        
    };
    
};


#endif /// APPS_IMPASTE_HH_