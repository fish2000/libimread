
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/objc-rt/objc-rt.hh>

#import "IMTestReceiver.h"

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

- (instancetype) callMethod {
    WTF("WTF output from within method:", ansi::lightcyan,
        "[imtsInstance callMethod]",      ansi::reset,
        "... dogg.");
    return self;
}

- (instancetype) callMethodWithInt:(int)arg {
    WTF("WTF output from within method:",               ansi::lightcyan,
        FF("[imtsInstance callMethodWithInt:%i]", arg), ansi::reset,
           "... dogg.");
    return self;
}

- (instancetype) callMethodWithInt:(int)arg andObjCString:(NSString *)anotherArg {
    const char *str = "nil";
    if (anotherArg != nil) { str = [anotherArg UTF8String]; }
    WTF("WTF output from within method:",                 ansi::lightcyan,
        FF("[imtsInstance callMethodWithInt:(%i)\n                          andObjCString:(%s)]",
            arg, str),
            ansi::reset, "... dogg.");
    return self;
}

- (instancetype) callMethodWithInt:(int)arg andVoidPointer:(void *)anotherVoidArg {
    const char *str = "nil";
    NSString *anotherArg = objc::bridge<NSString*>(anotherVoidArg);
    if (anotherArg != nil) { str = [anotherArg UTF8String]; }
    WTF("WTF output from within method:",                 ansi::lightcyan,
        FF("[imtsInstance callMethodWithInt:(%i)\n                          andVoidPointer:(%s)]",
            arg, str),
            ansi::reset, "... dogg.");
    return self;
}

- (float) returnFloat {
    WTF("WTF output from within method:", ansi::lightcyan,
        "[imtsInstance returnFloat]",     ansi::reset,
        "... dogg.");
    return 3.14159;
}

- (StructReturn) returnStruct {
    WTF("WTF output from within method:", ansi::lightcyan,
        "[imtsInstance returnStruct]",    ansi::reset,
        "... dogg.");
    StructReturn out;
    out.value = 666;
    return out;
}

@end
