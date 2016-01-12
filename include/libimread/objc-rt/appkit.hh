/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_OBJC_RT_APPKIT_HH
#define LIBIMREAD_OBJC_RT_APPKIT_HH

#ifdef __APPLE__
#import <AppKit/AppKit.h>
#endif /// __APPLE__

#include "types.hh"
#include "traits.hh"

namespace objc {
    
    namespace appkit {
        
        template <typename OCType>
        BOOL can_paste(NSPasteboard* board = nil) noexcept {
            static_assert(objc::traits::is_object<std::add_pointer_t<OCType>>::value,
                          "objc::appkit::can_paste<OCType> requires an Objective-C object type");
            if (!board) { board = [NSPasteboard generalPasteboard]; }
            return [board canReadObjectForClasses:@[ [OCType class] ]
                                          options:@{}];
        }
        
        template <typename OCType>
        __attribute__((ns_returns_retained))
        typename std::add_pointer_t<OCType> paste(NSPasteboard* board = nil) noexcept {
            static_assert(objc::traits::is_object<std::add_pointer_t<OCType>>::value,
                          "objc::appkit::paste<OCType> requires an Objective-C object type");
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
            static_assert(objc::traits::detail::are_object_pointers<OCTypes...>::value,
                          "objc::appkit::paste<OCType> requires an Objective-C object type");
            if (!board) { board = [NSPasteboard generalPasteboard]; }
            __attribute__((__unused__))
            NSInteger changecount = [board clearContents];
            return [board writeObjects:@[ objects... ]];
        }
        
        template <typename ...OCTypes> inline
        BOOL copy(OCTypes... objects) noexcept {
            static_assert(objc::traits::detail::are_object_pointers<OCTypes...>::value,
                          "objc::appkit::paste<OCType> requires an Objective-C object type");
            return copy_to<OCTypes...>([NSPasteboard generalPasteboard],
                                        objects...);
        }
        
    }
    
}

#endif /// LIBIMREAD_OBJC_RT_APPKIT_HH