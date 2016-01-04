/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef APPS_IMPASTE_HH_
#define APPS_IMPASTE_HH_

#include <type_traits>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#endif

#include <libimread/libimread.hpp>

/// printffery
#define IMPASTE_DEBUG 0

/// NSThread declarations and definitions,
/// one(ish) for each CLI option

@interface AXCheckThread : NSThread {}
@end

@interface AXDryRunThread : NSThread {}
@end

@interface AXImageSaveThread : NSThread {}
@end

/// boilerplate reduction strategem one:
/// macro + naming convention

#define AXTHREADRUN(threadName)                                                 \
    @autoreleasepool {                                                          \
        [NSApplication sharedApplication];                                      \
        [[[AX##threadName##Thread alloc] init] start];                          \
        [NSApp run];                                                            \
    };


namespace objc {
    
    /// boilerplate reduction strategem two:
    /// inlined function template
    
    template <typename OCThreadType> inline
    void run_thread(void) {
        @autoreleasepool {
            [NSApplication sharedApplication];
            [[[OCThreadType alloc] init] start];
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
        
        
    }
    
}


namespace im {
    
    namespace config {
        
        const std::string version = IM_VERSION;
        
    };
    
};


#endif /// APPS_IMPASTE_HH_