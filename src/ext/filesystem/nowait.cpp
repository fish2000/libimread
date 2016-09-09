
#include <libimread/ext/filesystem/nowait.h>

#ifdef IM_HAVE_AUTOFS_NOWAIT

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace filesystem {
    
    namespace detail {
        
        nowait_t::nowait_t()
            :descriptor(::open("/dev/autofs_nowait", 0))
            {}
        
        nowait_t::~nowait_t() {
            if (descriptor != -1) { ::close(descriptor); }
        }
        
        
    }
    
}

#endif
