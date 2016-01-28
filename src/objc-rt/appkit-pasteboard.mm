/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <libimread/libimread.hpp>
#include <libimread/objc-rt/appkit-pasteboard.hh>

namespace objc {
    
    namespace appkit {
        
        /// statically initialize the typemap and its inverse
        const PasteboardSubBase::typemap_t PasteboardSubBase::typemap = PasteboardSubBase::init_typemap();
        const PasteboardSubBase::typepam_t PasteboardSubBase::typepam = PasteboardSubBase::init_typepam();
        
    }
    
}
