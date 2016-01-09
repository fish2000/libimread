/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#import <AppKit/AppKit.h>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/IO/apple.hh>
#include <libimread/ext/categories/NSBitmapImageRep+IM.hh>
#include <libimread/ext/categories/NSData+IM.hh>
#include <libimread/ext/categories/NSURL+IM.hh>

namespace im {
    
    std::unique_ptr<Image> NSImageFormat::read(byte_source* src,
                                               ImageFactory* factory,
                                               options_map const& opts)  {
        @autoreleasepool {
            NSBitmapImageRep* rep = [NSBitmapImageRep imageRepWithByteVector:src->full_data()];
            std::unique_ptr<Image> output = [rep imageUsingImageFactory:factory];
            return output;
        }
    }
    
    void NSImageFormat::write(Image& input, byte_sink* output,
                                            options_map const& opts) {
        NSInteger filetype = objc::image::filetype(opts.get("extension"));
        imread_assert(filetype != -1,
                      "[apple-io] Can't write image of unknown type:",
                      opts.get("filename"));
        
        @autoreleasepool {
            NSBitmapImageRep* rep = [[NSBitmapImageRep alloc] initWithImage:input];
            NSData* datum = [rep representationUsingType:static_cast<NSBitmapImageFileType>(filetype)
                                              properties:@{}];
            [datum writeUsingByteSink:output];
        }
    }
    
}