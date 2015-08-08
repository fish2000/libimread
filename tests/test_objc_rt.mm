
#include <cstdlib>
#include <string>
#import <Cocoa/Cocoa.h>

#include <libimread/libimread.hpp>
#include <libimread/objc-rt.hh>
#include <libimread/errors.hh>
#include <libimread/fs.hh>

#include "include/catch.hpp"

@interface IMTestReceiver : NSObject {}
+ (void) callStatic;
+ (void) callStaticWithInt:(int)arg;
+ (void) callStaticWithInt:(int)arg andObjCString:(NSString*)anotherArg;
- (void) callMethod;
- (void) callMethodWithInt:(int)arg;
- (void) callMethodWithInt:(int)arg andObjCString:(NSString*)anotherArg;
@end

@implementation IMTestReceiver
+ (void) callStatic {
    WTF("WTF output from within function:",     ansi::lightcyan,
        "[IMTestReceiver callStatic]",          ansi::reset,
        "... dogg.");
}
+ (void) callStaticWithInt:(int)arg {
    WTF("WTF output from within function:",                 ansi::lightcyan,
        FF("[IMTestReceiver callStaticWithInt:%i]", arg),   ansi::reset,
        "... dogg.");
}
+ (void) callStaticWithInt:(int)arg andObjCString:(NSString*)anotherArg {
    WTF("WTF output from within function:",                 ansi::lightcyan,
        FF("[IMTestReceiver callStaticWithInt:(%i)", arg),  ansi::reset,
        FF("                    andObjCString:(%s)]",       [anotherArg UTF8String]),
           "... dogg.");
}
- (void) callMethod {
    WTF("WTF output from within method:", ansi::lightcyan,
        "[imtsInstance callMethod]",      ansi::reset,
        "... dogg.");
}
- (void) callMethodWithInt:(int)arg {
    WTF("WTF output from within method:",               ansi::lightcyan,
        FF("[imtsInstance callMethodWithInt:%i]", arg), ansi::reset,
           "... dogg.");
}
- (void) callMethodWithInt:(int)arg andObjCString:(NSString*)anotherArg {
    WTF("WTF output from within method:",                 ansi::lightcyan,
        FF("[imtsInstance callMethodWithInt:(%i)", arg),  ansi::reset,
        FF("                  andObjCString:(%s)]",       [anotherArg UTF8String]),
           "... dogg.");
}
@end


namespace {
    
    // TEST_CASE("[objc-rt] Call a static class function via objc::msg::send()",
    //           "[objc-rt-call-static-function]")
    // {
    //     objc::msg::send(
    //         objc::selector("callStatic"));
    // }
    
    TEST_CASE("[objc-rt] Call an instance method via objc::msg::send()",
              "[objc-rt-call-instance-method]")
    {
        @autoreleasepool {
            IMTestReceiver *imts = [[IMTestReceiver alloc] init];
            objc::msg::send(imts, objc::selector("callMethod"));
        }
    }
    
    TEST_CASE("[objc-rt] Call an instance method with an integer argument via objc::msg::send()",
              "[objc-rt-call-instance-method-one-arg-integer-value]")
    {
        @autoreleasepool {
            IMTestReceiver *imts = [[IMTestReceiver alloc] init];
            objc::msg::send(imts, objc::selector("callMethodWithInt:"), 42);
        }
    }
    
    TEST_CASE("[objc-rt] Call an instance method with int and NSString arguments via objc::msg::send()",
              "[objc-rt-call-instance-method-multiple-args-int-and-pointer-to-nsstring]")
    {
        @autoreleasepool {
            IMTestReceiver *imts = [[IMTestReceiver alloc] init];
            NSString __autoreleasing *stringArg = @"OH SHIT DOGG PARDON MY STRING PASSING";
            objc::msg::send(imts, objc::selector("callMethodWithInt:andObjCString:"), 42, stringArg);
        }
    }
    
    // BLEST_CASE("[objc-rt] Send a message via objc::msg::send()", "[objc-rt-msg-send]") {
    //     im::fs::NamedTemporaryFile temporary;
    //     NSData *datum;
    //     NSURL *url;
    //     NSString *urlpath;
    //     std::string path;
    //     //std::string prefix = "file://";
    //     std::string prefix = "";
    //     std::size_t nbytes = 20 * 1024; /// 20480
    //     unsigned char randos[20480] = {0};
    //
    //     arc4random_buf(static_cast<void*>(randos), nbytes);
    //     path = prefix + temporary.str();
    //
    //     @autoreleasepool {
    //         datum = [[NSData alloc] initWithBytes:(const void *)&randos[0]
    //                                        length:(NSInteger)nbytes];
    //         urlpath = [[NSString alloc] initWithUTF8String:path.c_str()];
    //         url = [[NSURL alloc] initWithString:urlpath];
    //
    //         // [datum writeToFile:urlpath atomically:YES];
    //
    //         // objc::msg::send((id)datum,
    //         //     objc::selector("writeToFile:atomically:"),
    //         //     urlpath, YES);
    //
    //         // objc_msgSend((id)datum,
    //         //     @selector(writeToFile:atomically:),
    //         //     urlpath, YES);
    //
    //         #if !__has_feature(objc_arc)
    //             [datum release];
    //             [urlpath release];
    //             [url release];
    //         #endif
    //     };
    //
    // }
    
}
