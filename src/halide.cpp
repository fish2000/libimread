
#include <libimread/libimread.hpp>
#include <libimread/halide.hh>

namespace im {
    
    void HalideBuffer::finalize() {}
    uint8_t *HalideBuffer::release(uint8_t *ptr) {
        // ptr_swap(ptr, allocation);
        // return ptr;
        return nullptr;
    }
    
    namespace halide {
        
        
        
    }
    

}