
#import <Cocoa/Cocoa.h>

@interface IMTestReceiver : NSObject {}
+ (void) callStatic;
+ (void) callStaticWithInt:(int)arg;
+ (void) callStaticWithInt:(int)arg andObjCString:(NSString *)anotherArg;
- (void) callMethod;
- (void) callMethodWithInt:(int)arg;
- (void) callMethodWithInt:(int)arg andObjCString:(NSString *)anotherArg;
- (void) callMethodWithInt:(int)arg andVoidPointer:(void *)anotherVoidArg;
@end
