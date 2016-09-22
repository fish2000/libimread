
#include <libimread/ext/filesystem/nowait.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef IM_HAVE_AUTOFS_NOWAIT

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

#ifdef IM_HAVE_AUTOFS_NOTRIGGER

namespace filesystem {
    
    namespace detail {
        
        notrigger_t::notrigger_t()
            :descriptor(::open("/dev/autofs_notrigger", 0))
            {}
        
        notrigger_t::~notrigger_t() {
            if (descriptor != -1) { ::close(descriptor); }
        }
        
    }
    
}

#endif
