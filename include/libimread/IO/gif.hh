/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_GIF_HH_
#define LIBIMREAD_IO_GIF_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/base.hh>
#include <libimread/ext/WriteGIF.hh>

namespace im {
    
    namespace detail {
        
        template <typename G>
        struct gifdisposer {
            constexpr gifdisposer() noexcept = default;
            template <typename U> gifdisposer(gifdisposer<U> const&) noexcept {};
            void operator()(G* gp) { gif::dispose(gp); gp = NULL; }
        };
        
        using gifholder = std::shared_ptr<gif::GIF>;
        
    }
    
    class GIFFormat : public ImageFormatBase<GIFFormat> {
        public:
            using can_write = std::true_type;
            using can_write_multi = std::true_type;
            
            DECLARE_OPTIONS(
                _signature = base64::encode("\x47\x49\x46\x38", 4),
                _siglength = 4,
                _suffix = "gif",
                _mimetype = "image/gif"
            );
            
            virtual void write(Image& input,
                               byte_sink* output,
                               const options_map& opts) override;
            
            virtual void write_multi(ImageList& input,
                                     byte_sink* output,
                                     const options_map& opts) override;
        
        private:
            void write_impl(Image& input, detail::gifholder g);
    };
    
    namespace format {
        using GIF = GIFFormat;
    }
    
}

/*
 *   [0]  http://www.astro.keele.ac.uk/oldusers/rno/Computing/File_magic.html
 */ 
#endif /// LIBIMREAD_IO_GIF_HH_
