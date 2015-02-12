
#include "halide.h"

namespace im {

    void HalideBuffer::finalize() {
        if (buffer.dev) { delete buffer.dev; }
        if (allocation != nullptr) { delete[] allocation; }
    }
    
    /*
    template <typename T>
    Image<T> imread(std::string filename) {
        HalideFactory<T> factory;
        im::options_map opts; /// not currently used when reading
        std::auto_ptr<im::ImageFormat> format(im::get_format(split_filename(filename.c_str())));
    
        _assert(format.get(), "[imread] Format is unknown to libimread\n");
        _assert(format->can_read(), "[imread] Format is unreadable by libimread\n");
    
        int fd = ::open(filename.c_str(), O_RDONLY | O_BINARY);
        _assert(!(fd < 0), "[imread] Filesystem/permissions error opening file\n");
    
        std::auto_ptr<im::byte_source> input(new im::fd_source_sink(fd));
        std::auto_ptr<im::Image> output = format->read(input.get(), &factory, opts);
    
        return static_cast<HalideImage<T> &>(*output).get();
    }
    */

}