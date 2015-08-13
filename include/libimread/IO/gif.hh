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
            template <typename U> gifdisposer(const gifdisposer<U>&) noexcept {};
            void operator()(G *gp) { if (gp) { gif::dispose(gp); gp = NULL; }}
        };
        
        using gifholder = std::shared_ptr<gif::GIF>;
        
    }
    
    class GIFFormat : public ImageFormat {
        public:
            typedef std::true_type can_write;
            typedef std::true_type can_write_multi;
            
            static bool match_format(byte_source *src) {
                /// 47 49 46 38 ("GIF8" in ASCII);
                /// ... from “File Magic Numbers” [0]
                return match_magic(src, "\x47\x49\x46\x38", 4);
            }
            
            virtual void write(Image &input,
                               byte_sink *output,
                               const options_map &opts) override;
            
            virtual void write_multi(ImageList &input,
                                     byte_sink* output,
                                     const options_map &opts) override;
        
        private:
            void write_impl(Image &&input, detail::gifholder &g);
    };
    
    namespace format {
        using GIF = GIFFormat;
    }
    
}

/*
 *   [0]  http://www.astro.keele.ac.uk/oldusers/rno/Computing/File_magic.html
 */ 
#endif /// LIBIMREAD_IO_GIF_HH_
