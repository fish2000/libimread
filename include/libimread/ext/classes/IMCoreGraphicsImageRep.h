#import <AppKit/NSImageRep.h>
#import <ApplicationServices/ApplicationServices.h>

#define OBJC_INITIALIZE                                     \
    do {                                                    \
        static BOOL hasBeenInitialized = NO;                \
        [super initialize];                                 \
        if (hasBeenInitialized) { return; }                 \
        hasBeenInitialized = YES;                           \
    } while (0);

inline CGRect CGRectWithRect(NSRect rect);
inline CGRect CGRectWithPointAndSize(NSPoint point, NSInteger width, NSInteger height);

@interface IMCoreGraphicsImageRep : NSImageRep {
    CGImageRef cgImage;
    NSString *colorSpaceName;
    id<NSObject> heldObject;
}

- initWithImageRef:(CGImageRef)myImage colorSpaceName:(NSString *)space;
- (void)setColorSpaceHolder:(id<NSObject>)anObject;
- (void)setImage:(CGImageRef)newImage;
@end
