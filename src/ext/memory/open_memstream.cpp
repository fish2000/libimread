/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)
/// Use funopen(3) to provide open_memstream(3) like functionality.

#include <cstring>
#include <cerrno>
#include <libimread/ext/memory/open_memstream.hh>

namespace memory {
    
    void memstream_grow(mstream* ms, std::size_t newsize) {
        char* buf;
        
        if (newsize > *ms->lenp) {
            buf = (char *)::realloc(*ms->cp, newsize + 1);
            if (buf != NULL) {
                std::memset(buf + *ms->lenp + 1, 0, newsize - *ms->lenp);
                *ms->cp = buf;
                *ms->lenp = newsize;
            }
        }
    }
    
    int memstream_read(void* cookie, char* buf, int len) {
        mstream* ms;
        int tocopy;
        
        ms = (mstream*)cookie;
        memstream_grow(ms, ms->offset + len);
        tocopy = *ms->lenp - ms->offset;
        if (len < tocopy) { tocopy = len; }
        
        std::memcpy(buf, *ms->cp + ms->offset, tocopy);
        ms->offset += tocopy;
        
        return tocopy;
    }
    
    int memstream_write(void* cookie, const char* buf, int len) {
        mstream* ms;
        int tocopy;
        
        ms = (mstream*)cookie;
        memstream_grow(ms, ms->offset + len);
        tocopy = *ms->lenp - ms->offset;
        if (len < tocopy) { tocopy = len; }
        
        std::memcpy(*ms->cp + ms->offset, buf, tocopy);
        ms->offset += tocopy;
        
        return tocopy;
    }
    
    fpos_t memstream_seek(void* cookie, fpos_t pos, int whence) {
        mstream* ms = (mstream*)cookie;
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
    
    int memstream_close(void* cookie) {
        ::free(cookie);
        return 0;
    }
    
} /// namespace memory


FILE* open_memstream(char** cp, std::size_t* lenp) {
    memory::mstream* ms;
    int save_errno;
    FILE* fp;
    *cp = NULL;
    *lenp = 0;
    ms = (memory::mstream*)::malloc(sizeof(*ms));
    ms->cp = cp;
    ms->lenp = lenp;
    ms->offset = 0;
    fp = ::funopen(ms,
        memory::memstream_read, memory::memstream_write,
        memory::memstream_seek, memory::memstream_close);
    if (fp == NULL) {
        save_errno = errno;
        ::free(ms);
        errno = save_errno;
    }
    return fp;
}