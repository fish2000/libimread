/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/gzio.hh>
#include <libimread/ext/filesystem/attributes.h>

/// Shortcut to std::string{ NULL_STR } value
#define STRINGNULL() store::stringmapper::base_t::null_value()

namespace im {
    
    namespace detail {
        using gzhandle_t = gzFile;
        
        char const* descriptor_mode(int descriptor) {
            switch (::fcntl(descriptor, F_GETFL, 0) & O_ACCMODE) {
                case O_RDONLY:  return "rb";
                case O_WRONLY:  return "wb";
                case O_RDWR:    return "rb"; /// gzopen doesn't know "r+"
                case O_APPEND:  return "ab"; /// this may not actually happen
                case O_CLOEXEC: return "WHAT THE HELL PEOPLE";
                default:        return "SERIOUSLY, FUCK FILE OPENMODES";
            }
        }
    }
    
    constexpr int gzio_source_sink::READ_FLAGS;
    constexpr int gzio_source_sink::WRITE_FLAGS;
    constexpr int gzio_source_sink::WRITE_CREATE_MASK;
    
    DECLARE_CONSTEXPR_CHAR(gzio_source_sink::kOriginalSize,     "im:original_size");
    DECLARE_CONSTEXPR_CHAR(gzio_source_sink::kUncompressedSize, "im:uncompressed_size");
    
    int gzio_source_sink::open_read(char const* p) const {
        return ::open(p, READ_FLAGS);
    }
    
    int gzio_source_sink::open_write(char const* p, int mask) const {
        return ::open(p, WRITE_FLAGS, mask);
    }
    
    gzio_source_sink::gzio_source_sink() {}
    
    gzio_source_sink::gzio_source_sink(detail::gzhandle_t gz)
        :gzhandle(gz)
        {}
    
    gzio_source_sink::gzio_source_sink(int fd)
        :descriptor(fd)
        ,gzhandle{ ::gzdopen(descriptor,
                     detail::descriptor_mode(descriptor)) }
        ,external(true)
        {}
    
    gzio_source_sink::~gzio_source_sink() { close(); }
    
    bool gzio_source_sink::can_seek() const noexcept { return true; } /// -ish (see SEEK_END note below)
    
    std::size_t gzio_source_sink::seek_absolute(std::size_t pos) { return ::gzseek(gzhandle, pos, SEEK_SET); }
    std::size_t gzio_source_sink::seek_relative(int delta) { return ::gzseek(gzhandle, delta, SEEK_CUR); }
    
    /// SEEK_END is, according to the zlib manual, UNSUPPORTED:
    std::size_t gzio_source_sink::seek_end(int delta) { return ::gzseek(gzhandle, delta, SEEK_END); }
    
    std::size_t gzio_source_sink::read(byte* buffer, std::size_t n) {
        int out = ::gzread(gzhandle, buffer, n);
        if (out == -1) {
            imread_raise(GZipIOError,
                "::gzread() returned -1",
                std::strerror(errno));
        }
        return static_cast<std::size_t>(out);
    }
    
    std::size_t gzio_source_sink::write(const void* buffer, std::size_t n) {
        int out = ::gzwrite(gzhandle, buffer, n);
        if (out == -1) {
            imread_raise(GZipIOError,
                "::gzwrite() returned -1",
                std::strerror(errno));
        }
        if (descriptor > 0 && out > 0) {
            filesystem::attribute::accessor_t accessor(descriptor, kUncompressedSize);
            std::size_t current_size = accessor.get() == STRINGNULL() ? 0 : std::stoul(accessor.get());
            std::string uncompressed_size = std::to_string(static_cast<std::size_t>(out) + current_size);
            if (!accessor.set(uncompressed_size)) {
                imread_raise(MetadataWriteError, "zlib xattr-storage failure:",
                    FF("\taccessor_t(%i, \"%s\")", descriptor, kUncompressedSize),
                    FF("\taccessor.set(\"%i\") returned false", std::stoi(uncompressed_size)));
            }
        }
        return static_cast<std::size_t>(out);
    }
    
    std::size_t gzio_source_sink::write(bytevec_t const& bv) {
        if (bv.empty()) { return 0; }
        return this->write(
            static_cast<const void*>(&bv[0]),
            bv.size());
    }
    
    detail::stat_t gzio_source_sink::stat() const {
        if (descriptor < 0) {
            imread_raise(FileSystemError,
                "cannot fstat(…) on an invalid descriptor");
        }
        detail::stat_t info;
        if (::fstat(descriptor, &info) == -1) {
            imread_raise(FileSystemError,
                "::fstat() returned -1",
                std::strerror(errno));
        }
        return info;
    }
    
    void gzio_source_sink::flush() { ::gzflush(gzhandle, Z_SYNC_FLUSH); }
    
    std::size_t gzio_source_sink::size() const {
        detail::stat_t info = this->stat();
        return info.st_size * sizeof(byte);
    }
    
    void* gzio_source_sink::readmap(std::size_t pageoffset) const {
        /// HOW IS COMPRES MMAP FORMED
        imread_raise_default(NotImplementedError);
    }
    
    std::string gzio_source_sink::xattr(std::string const& name) const {
        filesystem::attribute::accessor_t accessor(descriptor, name);
        return accessor.get();
    }
    
    std::string gzio_source_sink::xattr(std::string const& name, std::string const& value) const {
        filesystem::attribute::accessor_t accessor(descriptor, name);
        (value == STRINGNULL()) ? accessor.del() : accessor.set(value);
        return accessor.get();
    }
    
    int gzio_source_sink::xattrcount() const {
        return filesystem::attribute::fdcount(descriptor);
    }
    
    filesystem::detail::stringvec_t gzio_source_sink::xattrs() const {
        return filesystem::attribute::fdlist(descriptor);
    }
    
    std::size_t gzio_source_sink::original_byte_size() const {
        std::string out = this->xattr(kOriginalSize);
        return out == STRINGNULL() ? 0 : std::stoul(out);
    }
    
    std::size_t gzio_source_sink::original_byte_size(std::size_t new_size) const {
        std::string out = this->xattr(kOriginalSize, std::to_string(new_size));
        return out == STRINGNULL() ? 0 : std::stoul(out);
    }
    
    std::size_t gzio_source_sink::uncompressed_byte_size() const {
        std::string out = this->xattr(kUncompressedSize);
        return out == STRINGNULL() ? 0 : std::stoul(out);
    }
    
    std::size_t gzio_source_sink::uncompressed_byte_size(std::size_t new_size) const {
        std::string out = this->xattr(kUncompressedSize, std::to_string(new_size));
        return out == STRINGNULL() ? 0 : std::stoul(out);
    }
    
    float gzio_source_sink::compression_ratio() const {
        std::size_t compressed = size();
        std::size_t uncompressed = uncompressed_byte_size();
        if (compressed == 0 || uncompressed == 0) { return -1.0f; }
        return static_cast<float>(compressed) / static_cast<float>(uncompressed);
    }
    
    int gzio_source_sink::fd() const noexcept {
        return descriptor;
    }
    
    void gzio_source_sink::fd(int fd) noexcept {
        descriptor = fd;
        external = true;
    }
    
    bool gzio_source_sink::exists() const noexcept {
        if (descriptor < 0) { return false; }
        try {
            this->stat();
        } catch (FileSystemError&) {
            return false;
        }
        return true;
    }
    
    int gzio_source_sink::open(std::string const& spath, filesystem::mode fmode) {
        using filesystem::path;
        
        if (!path::exists(spath)) {
            /// change the default mode to write,
            /// if the file is nonexistant at the named path:
            fmode = filesystem::mode::WRITE;
        }
        
        if (fmode == filesystem::mode::WRITE) {
            descriptor = open_write(spath.c_str());
            if (descriptor < 0) {
                imread_raise(FileSystemError, "descriptor open-to-write failure:",
                    FF("\t::open(\"%s\", O_WRONLY | O_FSYNC | O_CREAT | O_EXCL | O_TRUNC)", spath.c_str()),
                    FF("\treturned negative value: %i", descriptor),
                       "\tERROR MESSAGE IS: ", std::strerror(errno));
            }
        } else {
            descriptor = open_read(spath.c_str());
            if (descriptor < 0) {
                imread_raise(FileSystemError, "descriptor open-to-read failure:",
                    FF("\t::open(\"%s\", O_RDONLY | O_FSYNC)", spath.c_str()),
                    FF("\treturned negative value: %i", descriptor),
                       "\tERROR MESSAGE IS: ", std::strerror(errno));
            }
        }
        
        /// the 'T' in the 'wT' mode string stands for 'transparent writes',
        /// according to the zlib manual (which is for append-only type situations)
        char const* modestring = fmode == filesystem::mode::READ ? "rb" : "wb";
        gzhandle = ::gzdopen(descriptor, modestring);
        
        if (!gzhandle) {
            imread_raise(GZipIOError, "zlib stream-open failure:",
                FF("\t::gzopen(\"%s\", %s)", spath.c_str(), modestring),
                FF("\treturned negative value: %i", descriptor),
                    "\tERROR MESSAGE IS: ", std::strerror(errno));
        }
        
        if (fmode == filesystem::mode::WRITE) {
            ::gzsetparams(gzhandle, 9, Z_FILTERED);
        }
        
        /// store original file size in xattr
        std::string original_size = std::to_string(path::filesize(spath));
        filesystem::attribute::accessor_t accessor(descriptor, kOriginalSize);
        if (!accessor.set(original_size)) {
            imread_raise(MetadataWriteError, "zlib xattr-storage failure:",
                FF("\taccessor_t(%i, \"%s\")", descriptor, kOriginalSize),
                FF("\taccessor.set(\"%i\") returned false", std::stoi(original_size)));
        }
        
        external = true;
        return descriptor;
    }
    
    int gzio_source_sink::close() {
        using std::swap;
        int out = -1;
        if (::gzclose(gzhandle) != Z_OK) {
            if (descriptor > 0) {
                ::close(descriptor);
            }
            imread_raise(GZipIOError, "error closing gzhandle",
                FF("\t::gzclose(%i)",   gzhandle),
                FF("\tdescriptor = %i", descriptor),
                "\tERROR MESSAGE IS: ", std::strerror(errno));
        }
        swap(out, descriptor);
        return out;
    }
    
    gzfile_source_sink::gzfile_source_sink(filesystem::mode fmode)
        :gzio_source_sink(), md(fmode)
        {}
    
    gzfile_source_sink::gzfile_source_sink(char* cpath, filesystem::mode fmode)
        :gzio_source_sink(), pth(cpath), md(fmode)
        {
            gzio_source_sink::open(const_cast<char const*>(cpath), fmode);
        }
    
    gzfile_source_sink::gzfile_source_sink(char const* ccpath, filesystem::mode fmode)
        :gzio_source_sink(), pth(ccpath), md(fmode)
        {
            gzio_source_sink::open(ccpath, fmode);
        }
    
    gzfile_source_sink::gzfile_source_sink(std::string& spath, filesystem::mode fmode)
        :gzio_source_sink(), pth(spath), md(fmode)
        {
            gzio_source_sink::open(spath, fmode);
        }
    
    gzfile_source_sink::gzfile_source_sink(std::string const& cspath, filesystem::mode fmode)
        :gzio_source_sink(), pth(cspath), md(fmode)
        {
            gzio_source_sink::open(cspath, fmode);
        }
    
    gzfile_source_sink::gzfile_source_sink(filesystem::path const& ppath, filesystem::mode fmode)
        :gzio_source_sink(), pth(ppath), md(fmode)
        {
            gzio_source_sink::open(pth.str(), fmode);
        }
    
    filesystem::path const& gzfile_source_sink::path() const {
        return pth;
    }
    
    bool gzfile_source_sink::exists() const noexcept {
        return pth.exists();
    }
    
    filesystem::mode gzfile_source_sink::mode(filesystem::mode m) {
        md = m;
        return md;
    }
    
    filesystem::mode gzfile_source_sink::mode() const {
        return md;
    }
    
    gzio::source::source()
        :gzfile_source_sink()
        {}
    
    gzio::source::source(char* cpath)
        :gzfile_source_sink(cpath)
        {}
    
    gzio::source::source(char const* ccpath)
        :gzfile_source_sink(ccpath)
        {}
    
    gzio::source::source(std::string& spath)
        :gzfile_source_sink(spath)
        {}
    
    gzio::source::source(std::string const& cspath)
        :gzfile_source_sink(cspath)
        {}
    
    gzio::source::source(filesystem::path const& ppath)
        :gzfile_source_sink(ppath)
        {}
    
    gzio::sink::sink()
        :gzfile_source_sink(filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(char* cpath)
        :gzfile_source_sink(cpath, filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(char const* ccpath)
        :gzfile_source_sink(ccpath, filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(std::string& spath)
        :gzfile_source_sink(spath, filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(std::string const& cspath)
        :gzfile_source_sink(cspath, filesystem::mode::WRITE)
        {}
    
    gzio::sink::sink(filesystem::path const& ppath)
        :gzfile_source_sink(ppath, filesystem::mode::WRITE)
        {}
    
}
