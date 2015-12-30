
#include <cstdlib>
#include <string>
#include <functional>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/objc-rt/objc-rt.hh>
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
        [imts release];
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
        [imts release];
    }
    
    TEST_CASE("[objc-rt] Call an instance method via objc::msg::get<Return, ...>() returning a float value",
              "[objc-rt-call-instance-method-return-float]")
    {
        @autoreleasepool {
            IMTestReceiver *imts = [[IMTestReceiver alloc] init];
            float out = objc::msg::get<float>(imts,
                objc::selector("returnFloat"));
            CHECK(out == 3.14159f);
        }
    }
    
    TEST_CASE("[objc-rt] Call an instance method via objc::msg::get<Return, ...>() returning a struct value",
              "[objc-rt-call-instance-method-return-struct]")
    {
        @autoreleasepool {
            IMTestReceiver *imts = [[IMTestReceiver alloc] init];
            StructReturn out = objc::msg::get<StructReturn>(imts,
                objc::selector("returnStruct"));
            CHECK(out.value == 666);
        }
    }
    
    TEST_CASE("[objc-rt] Confirm correct behavior of objc::boolean(bool) and objc::to_bool(BOOL)",
              "[objc-rt-confirm-behavior-objc-boolean-to_bool]")
    {
        /// objc::boolean(bool) -> BOOL;
        CHECK(objc::boolean(true) == YES);
        CHECK(objc::boolean(false) == NO);
        
        /// objc::to_bool(BOOL) -> bool;
        CHECK(objc::to_bool(YES) == true);
        CHECK(objc::to_bool(NO) == false);
    }
    
    TEST_CASE("[objc-rt] Test objc::id equality",
              "[objc-rt-test-objc-id-equality]")
    {
        NSString *st = @"Yo Dogg";
        NSString *so = @"I Heard You Like Hashed Comparable Objects";
        objc::id s(st);
        objc::id o(so);
        std::hash<objc::id> id_hasher;
        std::hash<objc::types::ID> object_hasher;
        
        bool check_one = bool(s == (id)st);
        bool check_two = bool(s == (id)@"Yo Dogg");
        bool check_three = bool(s != o);
        CHECK(check_one);
        CHECK(check_two);
        CHECK(check_three);
        CHECK(st != so);
        CHECK(s != o);
        
        /// check hashes via member methods
        CHECK(s.hash() == [st hash]);
        CHECK(s.hash() != [so hash]);
        
        /// check hashes via std::hash<T>
        CHECK(id_hasher(s) == object_hasher(st));
        CHECK(id_hasher(s) != object_hasher(so));
    }
    
    TEST_CASE("[objc-rt] Test objc::selector equality",
              "[objc-rt-test-objc-selector-equality]")
    {
        objc::selector yd("yoDogg:");
        objc::selector ih("iHeardYouLikeSelectors:");
        objc::selector s = @selector(yoDogg:);
        std::hash<objc::selector> struct_hasher;
        std::hash<objc::types::selector> type_hasher;
        
        bool check_one = bool(yd == s);
        bool check_two = bool(yd == @selector(yoDogg:));
        bool check_three = bool(yd != ih);
        CHECK(check_one);
        CHECK(check_two);
        CHECK(check_three);
        CHECK(yd == s);
        CHECK(yd != ih);
        
        /// check hashes via member methods
        CHECK(yd.hash() == s.hash());
        CHECK(yd.hash() != ih.hash());
        
        /// check hashes via std::hash<T>
        CHECK(struct_hasher(s) == struct_hasher(yd));
        CHECK(struct_hasher(s) != struct_hasher(ih));
        CHECK(struct_hasher(s) == type_hasher(@selector(yoDogg:)));
        CHECK(struct_hasher(s) != type_hasher(@selector(iHeardYouLikeSelectors:)));
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
        
        // WTF("Path variables:", prefixed, temporary.str());
        
        objc::msg::send(datum,
            objc::selector("writeToFile:atomically:"),
            filepath, YES);
        
        removed = temporary.remove();
        CHECK(removed == true);
        
        objc::msg::send(datum,
            objc::selector("writeToURL:atomically:"),
            url, YES);
        
        removed = temporary.remove();
        CHECK(removed == true);
        
        [datum release];
        [filepath release];
        [url release];
        
    }
    
}
