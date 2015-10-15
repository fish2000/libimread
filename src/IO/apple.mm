/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/IO/apple.hh>
#include <libimread/ext/categories/NSBitmapImageRep+IM.hh>

namespace im {
    
    std::unique_ptr<Image> NSImageFormat::read(byte_source *src,
                                               ImageFactory *factory,
                                               const options_map &opts)  {
        std::vector<byte> data = src->full_data();
        NSBitmapImageRep *rep = [NSBitmapImageRep imageRepWithByteVector:data];
        std::unique_ptr<Image> output = [rep imageUsingImageFactory:factory];
        #if !__has_feature(objc_arc)
            [rep release];
        #endif
        return output;
    }
    
}