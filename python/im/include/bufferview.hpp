/// Copyright 2012-2015 Alexander Bohn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_PYTHON_IM_INCLUDE_BUFFERVIEW_HPP_
#define LIBIMREAD_PYTHON_IM_INCLUDE_BUFFERVIEW_HPP_

#include <memory>
#include <libimread/image.hh>

/// forward-declare buffer_t and Py_buffer:
extern "C" {
    struct buffer_t;
    typedef struct bufferinfo Py_buffer;
}

/// forward-declare Halide::Type:
namespace Halide {
    struct Type;
}

namespace im {
    
    namespace buffer {
        
        /// forward-declare buffer heap deleter template:
        template <typename BufferType> struct deleter;
        
        /// forward-declare ViewFactory, for friendship:
        class ViewFactory;
        
        class View : public Image {
            
            public:
                using factory_t = buffer::ViewFactory;
                using shared_t = std::shared_ptr<buffer_t>;
                using allocation_t = std::unique_ptr<uint8_t[]>;
                using halotype_t = Halide::Type;
                
                View(halotype_t const&, int, int, int);
                View(View const&) noexcept;
                View(View&&) noexcept;
                View(buffer_t const*);
                View(Py_buffer const*);
                virtual ~View();
                
                /// Image API
                virtual uint8_t* data() const noexcept;
                virtual uint8_t* data(int) const;
                halotype_t type() const;
                buffer_t* buffer_ptr() const;
                virtual int nbits() const noexcept override;
                virtual int nbytes() const override;
                virtual int ndims() const override;
                virtual int dim(int) const noexcept override;
                virtual int stride(int) const noexcept override;
                virtual int min(int) const noexcept override;
                virtual bool is_signed() const noexcept override;
                virtual bool is_floating_point() const noexcept override;
                inline off_t rowp_stride() const noexcept;
                virtual void* rowp(int) const override;
                
            private:
                View() noexcept;
                shared_t shared;
                halotype_t htype;
                allocation_t allocation;
        };
        
#define xWIDTH d1
#define xHEIGHT d0
#define xDEPTH d2
        
        class ViewFactory : public ImageFactory {
            
            public:
                using image_t = buffer::View;
                using unique_t = std::unique_ptr<Image>;
                using shared_t = std::shared_ptr<Image>;
            
            private:
                std::string fname;
            
            public:
                ViewFactory() noexcept;
                ViewFactory(std::string const&) noexcept;
                virtual ~ViewFactory();
                
                std::string const& name() noexcept;
                std::string const& name(std::string const&) noexcept;
                
            protected:
                virtual unique_t create(int nbits,
                                        int xHEIGHT, int xWIDTH, int xDEPTH,
                                        int d3, int d4) override;
        };
        
#undef xWIDTH
#undef xHEIGHT
#undef xDEPTH
        
    }
    
} /* namespace im */

#endif /// LIBIMREAD_PYTHON_IM_INCLUDE_BUFFERVIEW_HPP_