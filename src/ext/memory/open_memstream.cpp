/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Use funopen(3) to provide open_memstream(3) like functionality.

#include <cstring>
#include <cerrno>
#include <libimread/ext/memory/open_memstream.hh>

namespace memory {
    
    namespace detail {
    
        void grow(mstream* ms, std::size_t newsize) {
            char* buf;
            if (newsize > *ms->lenp) {
                buf = (char*)std::realloc(*ms->cp, newsize + 1);
                if (buf != nullptr) {
                    std::memset(buf + *ms->lenp + 1, 0, newsize - *ms->lenp);
                    *ms->cp = buf;
                    *ms->lenp = newsize;
                }
            }
        }
        
    } /// namespace detail
    
    int read(void* cookie, char* buf, int len) {
        mstream* ms;
        int tocopy;
        
        ms = reinterpret_cast<mstream*>(cookie);
        detail::grow(ms, ms->offset + len);
        tocopy = *ms->lenp - ms->offset;
        if (len < tocopy) { tocopy = len; }
        
        std::memcpy(buf, *ms->cp + ms->offset, tocopy);
        ms->offset += tocopy;
        return tocopy;
    }
    
    int write(void* cookie, const char* buf, int len) {
        mstream* ms;
        int tocopy;
        
        ms = reinterpret_cast<mstream*>(cookie);
        detail::grow(ms, ms->offset + len);
        tocopy = *ms->lenp - ms->offset;
        if (len < tocopy) { tocopy = len; }
        
        std::memcpy(*ms->cp + ms->offset, buf, tocopy);
        ms->offset += tocopy;
        return tocopy;
    }
    
    fpos_t seek(void* cookie, fpos_t pos, int whence) {
        mstream* ms = reinterpret_cast<mstream*>(cookie);
        switch (whence) {
            case SEEK_SET:
                ms->offset = pos;
                break;
            case SEEK_CUR:
                ms->offset += pos;
                break;
            case SEEK_END:
                ms->offset = *ms->lenp + pos;
                break;
        }
        return ms->offset;
    }
    
    int close(void* cookie) {
        std::free(cookie);
        return 0;
    }
    
} /// namespace memory


FILE* open_memstream(char** cp, std::size_t* lenp) {
    memory::mstream* ms;
    FILE* filepointer;
    
    *cp = nullptr;
    *lenp = 0;
    
    ms = reinterpret_cast<memory::mstream*>(std::malloc(sizeof(*ms)));
    ms->cp = cp;
    ms->lenp = lenp;
    ms->offset = 0;
    
    filepointer = ::funopen(ms,
        memory::read, memory::write,
        memory::seek, memory::close);
    
    if (filepointer == nullptr) {
        int saved_errno = errno;
        std::free(ms);
        errno = saved_errno;
    }
    
    return filepointer;
}