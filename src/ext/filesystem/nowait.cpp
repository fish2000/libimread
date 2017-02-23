
#include <libimread/ext/filesystem/nowait.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef IM_HAVE_AUTOFS_NOWAIT

namespace filesystem {
    
    namespace detail {
        
        std::atomic<int> nowait_t::descriptor{ -1 };
        std::atomic<int> nowait_t::retaincount{ 0 };
        
        nowait_t::nowait_t() {
            if (retaincount.fetch_add(1) < 1) {
                descriptor.store(::open("/dev/autofs_nowait", 0));
            }
        }
        
        nowait_t::~nowait_t() {
            if (retaincount.fetch_sub(1) == 1) {
                ::close(descriptor.load());
            }
        }
        
    }
    
}

#endif

#ifdef IM_HAVE_AUTOFS_NOTRIGGER

namespace filesystem {
    
    namespace detail {
        
        std::atomic<int> notrigger_t::descriptor{ -1 };
        std::atomic<int> notrigger_t::retaincount{ 0 };
        
        notrigger_t::notrigger_t() {
            if (retaincount.fetch_add(1) < 1) {
                descriptor.store(::open("/dev/autofs_notrigger", 0));
            }
        }
        
        notrigger_t::~notrigger_t() {
            if (retaincount.fetch_sub(1) == 1) {
                ::close(descriptor.load());
            }
        }
        
    }
    
}

#endif
