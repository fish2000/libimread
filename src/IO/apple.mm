/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/IO/apple.hh>
#include <libimread/ext/categories/NSBitmapImageRep+IM.hh>

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
    
}