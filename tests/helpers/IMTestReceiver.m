
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/objc-rt.hh>

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
- (void) callMethodWithInt:(int)arg andObjCString:(NSString *)anotherArg {
    const char *str = "nil";
    if (anotherArg != nil) { str = [anotherArg UTF8String]; }
    WTF("WTF output from within method:",                 ansi::lightcyan,
        FF("[imtsInstance callMethodWithInt:(%i)\n                          andObjCString:(%s)]",
            arg, str),
            ansi::reset, "... dogg.");
}
- (void) callMethodWithInt:(int)arg andVoidPointer:(void *)anotherVoidArg {
    const char *str = "nil";
    NSString *anotherArg = (__bridge NSString*)anotherVoidArg;
    if (anotherArg != nil) { str = [anotherArg UTF8String]; }
    WTF("WTF output from within method:",                 ansi::lightcyan,
        FF("[imtsInstance callMethodWithInt:(%i)\n                          andVoidPointer:(%s)]",
            arg, str),
            ansi::reset, "... dogg.");
}
@end
