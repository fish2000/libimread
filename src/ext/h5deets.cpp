
#include <libimread/ext/h5deets.hh>

namespace im {
    
    namespace detail {
        
        /// NO-Op h5base releaser function:
        const h5base::releaser_f h5base::NOOp = [](hid_t hid) -> herr_t { return -1; };
        
    } /* namespace detail */
    
} /* namespace im */