
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HH_
#define LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HH_

#include <memory>
#include <Python.h>

namespace im {
    
    namespace detail {
        
        /// XXX: remind me why in fuck did I write this shit originally
        template <typename T, typename pT>
        std::unique_ptr<T> dynamic_cast_unique(std::unique_ptr<pT>&& source) {
            /// Force a dynamic_cast upon a unique_ptr via interim swap
            /// ... danger, will robinson: DELETERS/ALLOCATORS NOT WELCOME
            /// ... from http://stackoverflow.com/a/14777419/298171
            if (!source) { return std::unique_ptr<T>(); }
            
            /// Throws a std::bad_cast() if this doesn't work out
            T *destination = &dynamic_cast<T&>(*source.get());
            
            source.release();
            std::unique_ptr<T> out(destination);
            return out;
        }
        
        PyObject* structcode_to_dtype(char const* code);
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HH_