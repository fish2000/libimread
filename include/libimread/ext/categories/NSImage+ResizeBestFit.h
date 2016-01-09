//
//  NSImage+Resize.h
//  PhotoNoteBook
//
//  Created by 안 진섭 on 6/11/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>

@interface NSImage (ResizeImageBestFit)

- (NSImage*) imageByBestFitForSize:(CGSize)targetSize;
- (NSImage*) imageByScalingAndCroppingForSize:(CGSize)targetSize;
- (NSData*)  PNGData;
- (NSData*)  JPEGData;
- (NSData*)  JPEGDataWithCompression:(float)factor;

@end
