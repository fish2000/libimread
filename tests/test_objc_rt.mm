
#include <cstdlib>
#include <string>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/objc-rt.hh>
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
            objc::msg::send(imts,
                objc::selector("callMethod"));
        }
    }
    
    TEST_CASE("[objc-rt] Call an instance method with an integer argument via objc::msg::send()",
              "[objc-rt-call-instance-method-one-arg-integer-value]")
    {
        @autoreleasepool {
            IMTestReceiver *imts = [[IMTestReceiver alloc] init];
            objc::msg::send(imts,
                objc::selector("callMethodWithInt:"), 42);
        }
    }
    
    TEST_CASE("[objc-rt] Call an instance method with int and NSString arguments via objc::msg::send()",
              "[objc-rt-call-instance-method-multiple-args-int-and-pointer-to-nsstring]")
    {
        IMTestReceiver *imts = [[IMTestReceiver alloc] init];
        NSString *stringArg = @"OH SHIT DOGG PARDON MY STRING PASSING";
        [stringArg retain];
        objc::msg::send(imts,
            objc::selector("callMethodWithInt:andObjCString:"),
            42, stringArg);
        [stringArg release];
    }
    
    TEST_CASE("[objc-rt] Call an instance method with int and (__bridge void*)NSString arguments via objc::msg::send()",
              "[objc-rt-call-instance-method-multiple-args-int-and-pointer-to-nsstring]")
    {
        IMTestReceiver *imts = [[IMTestReceiver alloc] init];
        NSString *stringArg = @"OH SHIT DOGG PARDON MY STRING PASSING";
        [stringArg retain];
        objc::msg::send(imts,
            objc::selector("callMethodWithInt:andVoidPointer:"),
            42, objc::bridge<void*>(stringArg));
        [stringArg release];
    }
    
    TEST_CASE("[objc-rt] Test objc::id equality",
              "[objc-rt-test-objc-id-equality]")
    {
        NSString *st = @"Yo Dogg";
        NSString *so = @"I Heard You Like Hashed Comparable Objects";
        objc::id s(st);
        objc::id o(so);
        //REQUIRE(s == st);
        //REQUIRE(s == @"Yo Dogg");
        bool check_one = bool(s == (id)st);
        bool check_two = bool(s == (id)@"Yo Dogg");
        bool check_three = bool(s != o);
        REQUIRE(check_one);
        REQUIRE(check_two);
        REQUIRE(check_three);
    }
    
    TEST_CASE("[objc-rt] Send a message via objc::msg::send()", "[objc-rt-msg-send]") {
        im::fs::NamedTemporaryFile temporary;
        NSData *datum;
        NSString *filepath;
        NSURL *url;
        
        std::string prefix = "file://";
        std::size_t nbytes = 20 * 1024; /// 20480
        unsigned char randos[20480] = {0};
        bool removed = false;
        
        arc4random_buf(static_cast<void*>(randos), nbytes);
        std::string prefixed = prefix + temporary.str();
        
        datum = [[NSData alloc] initWithBytes:(const void *)&randos[0]
                                    length:(NSInteger)nbytes];
        filepath = [[NSString alloc] initWithUTF8String:temporary.c_str()];
        url = [[NSURL alloc] initWithString:[
            [NSString alloc] initWithUTF8String:prefixed.c_str()]];
        
        [datum retain];
        [filepath retain];
        [url retain];
        
        WTF("Path variables:", prefixed, temporary.str());
        
        objc::msg::send((id)datum,
            objc::selector("writeToFile:atomically:"),
            filepath, YES);
        
        removed = temporary.remove();
        REQUIRE(removed == true);
        
        objc::msg::send((id)datum,
            objc::selector("writeToURL:atomically:"),
            url, YES);
        
        removed = temporary.remove();
        REQUIRE(removed == true);
        
        [datum release];
        [filepath release];
        [url release];
        
    }
    
}
