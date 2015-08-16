
#include <cstdlib>
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/objc-rt.hh>
#include <libimread/errors.hh>
#include <libimread/fs.hh>

#include "include/catch.hpp"
#import "helpers/IMTestReceiver.h"

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
        IMTestReceiver *imts = [[IMTestReceiver alloc] init];
        NSString *stringArg = @"OH SHIT DOGG PARDON MY STRING PASSING";
        [stringArg retain];
        objc::msg::send(imts, objc::selector("callMethodWithInt:andObjCString:"), 42, stringArg);
        [stringArg release];
    }
    
    TEST_CASE("[objc-rt] Call an instance method with int and (__bridge void*)NSString arguments via objc::msg::send()",
              "[objc-rt-call-instance-method-multiple-args-int-and-pointer-to-nsstring]")
    {
        IMTestReceiver *imts = [[IMTestReceiver alloc] init];
        NSString *stringArg = @"OH SHIT DOGG PARDON MY STRING PASSING";
        [stringArg retain];
        objc::msg::send(imts, objc::selector("callMethodWithInt:andVoidPointer:"), 42, objc::bridge<void*>(stringArg));
        [stringArg release];
    }
    
    TEST_CASE("[objc-rt] Send a message via objc::msg::send()", "[objc-rt-msg-send]") {
        im::fs::NamedTemporaryFile temporary;
        NSData *datum;
        NSURL *url;
        NSString *urlpath;
        std::string path;
        //std::string prefix = "file://";
        std::string prefix = "";
        std::size_t nbytes = 20 * 1024; /// 20480
        unsigned char randos[20480] = {0};
        
        arc4random_buf(static_cast<void*>(randos), nbytes);
        path = prefix + temporary.str();
        
        @autoreleasepool {
            datum = [[NSData alloc] initWithBytes:(const void *)&randos[0]
                                           length:(NSInteger)nbytes];
            urlpath = [[NSString alloc] initWithUTF8String:path.c_str()];
            url = [[NSURL alloc] initWithString:urlpath];
            
            [datum retain];
            [urlpath retain];
            
            objc::msg::send((id)datum,
                objc::selector("writeToFile:atomically:"),
                urlpath, YES);
            
            [datum release];
            [urlpath release];
            [url release];
        };
        
    }
    
}
