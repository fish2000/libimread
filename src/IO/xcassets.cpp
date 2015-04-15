// Copyright 2012 Luis Pedro Coelho <luis@luispedro.org>
// License: MIT (see COPYING.MIT file)

#include <libimread/IO/xcassets.hh>

namespace im {
    
    void XCAssetsFormat::write(Image &input,
            byte_sink *output,
            const options_map &opts)
    {
        
        /// Write the image as a PNG into a new byte_sink
        // byte_sink png_sink()
        PNGFormat::write(input, output, opts);
        
    }
    
}