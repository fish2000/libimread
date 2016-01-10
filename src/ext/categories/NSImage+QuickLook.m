//
//  NSImage+QuickLook.m
//  QuickLookTest
//
//  Created by Matt Gemmell on 29/10/2007.
//

#import <libimread/ext/categories/NSImage+QuickLook.h>
#import <QuickLook/QuickLook.h> // Remember to import the QuickLook framework into your project!

@implementation NSImage (QuickLook)

+ (instancetype) imageWithPreviewOfFileAtPath:(NSString *)path
                                       ofSize:(NSSize)size
                                       asIcon:(BOOL)icon {
    
    NSURL* fileURL = [NSURL fileURLWithPath:path];
    if (!path || !fileURL) { return nil; }
    
    CGImageRef ref = QLThumbnailImageCreate(
        kCFAllocatorDefault,
        (CFURLRef)fileURL,
        CGSizeMake(size.width, size.height),
        (CFDictionaryRef)@{ 
            (NSString*)kQLThumbnailOptionIconModeKey : [NSNumber numberWithBool:icon]
        }
    );
    
    if (ref == NULL) {
        /// If we couldn't get a Quick Look preview,
        /// fall back on the file's Finder icon.
        NSImage* iicon = [[NSWorkspace sharedWorkspace] iconForFile:path];
        if (iicon) { [iicon setSize:size]; }
        return iicon;
    }
    
    // Take advantage of NSBitmapImageRep's -initWithCGImage: initializer, new in Leopard,
    // which is a lot more efficient than copying pixel data into a brand new NSImage.
    // Thanks to Troy Stephens @ Apple for pointing this new method out to me.
    NSBitmapImageRep* bitmap = [[NSBitmapImageRep alloc] initWithCGImage:ref];
    NSImage* newImage = nil;
    
    if (bitmap) {
        newImage = [[NSImage alloc] initWithSize:[bitmap size]];
        if (newImage) {
            [newImage addRepresentation:bitmap];
            CFRelease(ref);
            return newImage;
        }
    }
    
    CFRelease(ref);
    return nil;
}


@end
