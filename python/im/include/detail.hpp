
#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_

#include <memory>
#include <vector>
#include <string>
#include <Python.h>

namespace py {
    
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
        
        using stringvec_t = std::vector<std::string>;
        stringvec_t& formats_as_vector();                   /// this one is GIL-optional (how european!)
        PyObject* formats_as_pytuple(int idx = 0);          /// whereas here, no GIL no shoes no funcall
        
    }
    
}

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_DETAIL_HPP_