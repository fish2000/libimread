/// Copyright 2014 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#ifndef LIBIMREAD_IO_GIF_HH_
#define LIBIMREAD_IO_GIF_HH_

#include <memory>
#include <libimread/libimread.hpp>
#include <libimread/errors.hh>
#include <libimread/image.hh>
#include <libimread/imageformat.hh>
#include <libimread/imagelist.hh>
#include <libimread/metadata.hh>
#include <libimread/ext/WriteGIF.hh>

namespace im {
    
    namespace detail {
        
        /// Shared pointer, holding a pointer to gif::GIF:
        using gifholder = std::shared_ptr<gif::GIF>;
        
        /// Forward declaration --
        /// GIF Image pixel-data buffer-babysitter structure:
        struct gifbuffer;
        
        /// Deleter functor template for gif::GIF data --
        /// For use with std::unique_ptr or std::shared_ptr:
        template <typename G>
        struct gifdisposer {
            constexpr gifdisposer() noexcept = default;
            template <typename U> gifdisposer(gifdisposer<U> const&) noexcept {};
            void operator()(std::add_pointer_t<G> gp) {
                gif::dispose(gp);
                gp = nullptr; 
            }
        };
        
        /// “gifholder” shared-pointer factory function --
        /// calls pseudo-constructor gif::newGIF():
        gifholder gifsink(int frame_interval = 3);
    }
    
    class GIFFormat : public ImageFormatBase<GIFFormat> {
        
        public:
            
            #if defined(__APPLE__)
            using can_read = std::true_type;
            using can_read_multi = std::true_type;
            using can_read_metadata = std::true_type;
            #endif
            using can_write = std::true_type;
            using can_write_multi = std::true_type;
            
            DECLARE_OPTIONS(
                _signatures = {
                    SIGNATURE("\x47\x49\x46\x38", 4)
                },
                _suffixes = { "gif" },
                _mimetype = "image/gif",
                _delay = 3      /// default delay between animated GIF frames,
                                /// … I have no idea what unit that is in, btw
            );
            
            virtual std::unique_ptr<Image> read(byte_source* src,
                                                ImageFactory* factory,
                                                Options const& opts) override {
                #if defined(__APPLE__)
                    ImageList pages = read_impl(src, factory, false, opts);
                    std::unique_ptr<Image> out = pages.pop();
                    return out;
                #endif
                imread_raise_default(NotImplementedError);
            }
            
            virtual ImageList read_multi(byte_source* src,
                                         ImageFactory* factory,
                                         Options const& opts) override {
                #if defined(__APPLE__)
                    return read_impl(src, factory, true, opts);
                #endif
                imread_raise_default(NotImplementedError);
            }
            
            virtual Metadata read_metadata(byte_source* src,
                                           Options const& opts) override {
                #if defined(__APPLE__)
                    return read_metadata_impl(src, opts);
                #endif
                imread_raise_default(NotImplementedError);
            }
            
            virtual void write(Image& input,
                               byte_sink* output,
                               Options const& opts) override;
            
            virtual void write_multi(ImageList& input,
                                     byte_sink* output,
                                     Options const& opts) override;
            
        private:
            
            void        write_impl(Image const& input, detail::gifholder& g,
                                                       detail::gifbuffer& gbuf,
                                                       int framedelay = -1);
            
            #if defined(__APPLE__)
            ImageList   read_impl(byte_source* src,
                                  ImageFactory* factory,
                                  bool is_multi,
                                  Options const& opts);
            
            Metadata    read_metadata_impl(byte_source* src, Options const& opts);
            #endif
    };
    
    namespace format {
        using GIF = GIFFormat;
    }
    
}

/*
 *   [0]  http://www.astro.keele.ac.uk/oldusers/rno/Computing/File_magic.html
 */ 
#endif /// LIBIMREAD_IO_GIF_HH_
