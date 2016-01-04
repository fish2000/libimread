// Copyright 2001-2005, 2010-2013 Omni Development, Inc. All rights reserved.
//
// I, Alexander Bohn, totally ripped this class off from these talented people:
//      http://git.io/vc7tO
//
// This software may only be used and reproduced according to the
// terms in the file OmniSourceLicense.html, which should be
// distributed with this project and can also be found at
// <http://www.omnigroup.com/developer/sourcecode/sourcelicense/>.

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

@interface AXCoreGraphicsImageRep : NSImageRep {
    CGImageRef cgImage;
    NSString* colorSpaceName;
    id<NSObject> heldObject;
}

- initWithImageRef:(CGImageRef)myImage colorSpaceName:(NSString*)space;
- (void)setColorSpaceHolder:(id<NSObject>)anObject;
- (void)setImage:(CGImageRef)newImage;
@end
