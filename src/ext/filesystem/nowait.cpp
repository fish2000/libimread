
#include <libimread/ext/filesystem/nowait.h>

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace filesystem {
    
    namespace detail {
        
        #ifdef IM_HAVE_AUTOFS_NOWAIT
        VFS_INHIBITOR_DEFINITION(nowait_t, "/dev/autofs_nowait");
        #endif
        
        #ifdef IM_HAVE_AUTOFS_NOTRIGGER
        VFS_INHIBITOR_DEFINITION(notrigger_t, "/dev/autofs_notrigger");
        #endif
        
    }
    
}