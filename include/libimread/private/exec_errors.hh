std::string error_message(int error) {
    switch (error) {
    #ifdef EACCES
        case EACCES:        return "[EACCES] Permission denied";
    #endif
    #ifdef EPERM
        case EPERM:         return "[EPERM] Superuser permission is required";
    #endif
    #ifdef E2BIG
        case E2BIG:         return "[E2BIG] Argument list too long";
    #endif
    #ifdef ENOEXEC
        case ENOEXEC:       return "[ENOEXEC] Exec format error";
    #endif
    #ifdef EFAULT
        case EFAULT:        return "[EFAULT] Invalid address";
    #endif
    #ifdef ENAMETOOLONG
        case ENAMETOOLONG:  return "[ENAMETOOLONG] A path name is too long";
    #endif
    #ifdef ENOENT
        case ENOENT:        return "[ENOENT] No such file or directory";
    #endif
    #ifdef ENOMEM
        case ENOMEM:        return "[ENOMEM] Not enough core memory";
    #endif
    #ifdef ENOTDIR
        case ENOTDIR:       return "[ENOTDIR] Not a directory";
    #endif
    #ifdef ELOOP
        case ELOOP:         return "[ELOOP] Number of symbolic links exceeds system limit";
    #endif
    #ifdef ETXTBSY
        case ETXTBSY:       return "[ETXTBSY] Text file is in use";
    #endif
    #ifdef EIO
        case EIO:           return "[EIO] I/O error";
    #endif
    #ifdef ENFILE
        case ENFILE:        return "[ENFILE] Number of open files exceeds system limit";
    #endif
    #ifdef EINVAL
        case EINVAL:        return "[EINVAL] Invalid argument";
    #endif
    #ifdef EISDIR
        case EISDIR:        return "[EISDIR] Is a directory";
    #endif
    #ifdef ELIBBAD
        case ELIBBAD:       return "[ELIBBAD] Shared library file is invalid or corrupt";
    #endif
        default: {
            if (error) {
                std::string message(std::strerror(error));
                return message;
            }
        }
    }
}