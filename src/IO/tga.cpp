/// Copyright 2016 Alexander Böhn <fish2000@gmail.com>
/// License: MIT (see COPYING.MIT file)

#include <iod/json.hh>
#include <libimread/errors.hh>
#include <libimread/IO/tga.hh>
#include <libimread/seekable.hh>
#include <libimread/options.hh>

namespace im {
    
    DECLARE_FORMAT_OPTIONS(TGAFormat);
    
    namespace detail { /// formerly 'TGA_pvt'
        
        enum tga_image_type {
            TYPE_NODATA         = 0,    ///< image with no data (why even spec it?)
            TYPE_PALETTED       = 1,    ///< paletted RGB
            TYPE_RGB            = 2,    ///< can include alpha
            TYPE_GRAY           = 3,    ///< can include alpha
            TYPE_PALETTED_RLE   = 9,    ///< same as PALETTED but run-length encoded
            TYPE_RGB_RLE        = 10,   ///< same as RGB but run-length encoded
            TYPE_GRAY_RLE       = 11    ///< same as GRAY but run-length encoded
        };
        
        enum tga_flags {
            FLAG_X_FLIP     = 0x10,   ///< right-left image
            FLAG_Y_FLIP     = 0x20    ///< top-down image
        };
        
        /// Targa file header:
        
        typedef struct {
            uint8_t idlen;              ///< image comment length
            uint8_t cmap_type;          ///< palette type
            uint8_t type;               ///< image type (see tga_image_type)
            uint16_t cmap_first;        ///< offset to first entry
            uint16_t cmap_length;       ///<
            uint8_t cmap_size;          ///< palette size
            uint16_t x_origin;          ///<
            uint16_t y_origin;          ///<
            uint16_t width;             ///< image width
            uint16_t height;            ///< image height
            uint8_t bpp;                ///< bits per pixel
            uint8_t attr;               ///< attribs (alpha bits and \ref tga_flags)
        } tga_header;
        
        /// TGA 2.0 file footer:
        
        typedef struct {
            uint32_t ofs_ext;           ///< offset to the extension area
            uint32_t ofs_dev;           ///< offset to the developer directory
            char signature[18];         ///< file signature string
        } tga_footer;
        
        /// TGA 2.0 developer directory entry:
        
        typedef struct {
            uint16_t tag;               ///< tag
            uint32_t ofs;               ///< byte offset to the tag data
            uint32_t size;              ///< tag data length
        } tga_devdir_tag;
        
        /// ... this is used in the extension area:
        
        enum tga_alpha_type {
            TGA_ALPHA_NONE              = 0,    ///< no alpha data included
            TGA_ALPHA_UNDEFINED_IGNORE  = 1,    ///< can ignore alpha
            TGA_ALPHA_UNDEFINED_RETAIN  = 2,    ///< undefined, but should be retained
            TGA_ALPHA_USEFUL            = 3,    ///< useful alpha data is present
            TGA_ALPHA_PREMULTIPLIED     = 4     ///< alpha is pre-multiplied (arrrgh!)
            // values 5-127 are reserved
            // values 128-255 are unassigned
        };
        
        /// Helper - write, with error detection (no byte swapping!)
        template <typename T>
        bool write(T const* buffer, byte_sink*  output,
                                    std::size_t itemsize = sizeof(T),
                                    std::size_t nitems = 1) {
            if (itemsize * nitems == 0) { return true; }
            std::size_t n = output->write(buffer, itemsize * nitems);
            if (n != nitems) {
                /// error ("Write error:
                ///         wrote %d records of %d", (int)n, (int)nitems);
                imread_raise(CannotWriteError,
                    "Proxied write error");
                // return false;
            }
            return true;
        }
        
        /// Helper -- write a 'short' with byte swapping if necessary
        bool write(uint16_t s, byte_sink* output) {
            if (detail::bigendian()) {
                detail::swap_endian(&s);
            }
            return write(&s, output, sizeof(s), 1);
        }
        
        bool write(uint32_t i, byte_sink* output) {
            if (detail::bigendian()) {
                detail::swap_endian(&i);
            }
            return write(&i, output, sizeof(i), 1);
        }
        
        bool pad(byte_sink* output, std::size_t n = 1) {
            if (n == 0) { return false; }
            char zeros[n] = { 0 };
            return output->write(zeros, n) == n;
        }
        
        bool paddedwrite(std::string const& s, std::size_t len, byte_sink* output) {
            std::size_t slen = std::min(s.length(), len - 1);
            return write(s.c_str(), output, slen) && pad(output, len-slen);
        }
        
    }  /* namespace detail */
    
    
    std::unique_ptr<Image> TGAFormat::read(byte_source* src, ImageFactory* factory,
                                           Options const& opts) {
        
        std::unique_ptr<Image> output = factory->create(bit_depth, height, width, channels);
        return output;
    }
    
    
    void TGAFormat::write(Image& input, byte_sink* output,
                          Options const& opts) {
        /// DO IT DOUG
        bool dither = opts.cast<bool>("tga:dither", false);
        bool use_rle = opts.cast<bool>("tga:compress", false);
        float gamma_value = opts.cast<bool>("tga:gamma", 1.0);
        std::string targa_id = opts.cast<std::string>("tga:id", "");
        std::size_t targa_idlen = 0;
        
        /// prepare Targa header struct
        detail::tga_header tga = { 0 };
        tga.type = input.planes() <= 2 ? detail::tga_image_type::TYPE_GRAY :
                               use_rle ? detail::tga_image_type::TYPE_RGB_RLE :
                                         detail::tga_image_type::TYPE_RGB;
        tga.bpp = input.planes() * 8;
        tga.width = input.width();
        tga.height = input.height();
        tga.x_origin = 0; /// GENERALLY UNUSED
        tga.y_origin = 0; /// GENERALLY UNUSED
        
        tga.idlen = std::min(targa_id.size(), (std::size_t)255);
        targa_idlen = tga.idlen;
        
        /// This next line is a fucking sweet OIIO one-liner trick:
        if (input.planes() % 2 == 0) {
                                /// WE HAVE ALPHA:
            tga.attr = 8;       /// 8 bits of alpha
        }
        /// Add Y_FLIP flag if we're using RLE:
        if (use_rle) {
            tga.attr |= detail::tga_flags::FLAG_Y_FLIP;
        }
        
        /// TGAs are little-endian, soooo:
        if (detail::bigendian()) {
            detail::swap_endian(&tga.cmap_type);
            detail::swap_endian(&tga.type);
            detail::swap_endian(&tga.cmap_first);
            detail::swap_endian(&tga.cmap_length);
            detail::swap_endian(&tga.cmap_size);
            detail::swap_endian(&tga.x_origin);
            detail::swap_endian(&tga.y_origin);
            detail::swap_endian(&tga.width);
            detail::swap_endian(&tga.height);
            detail::swap_endian(&tga.bpp);
            detail::swap_endian(&tga.attr);
        }
        
        /// OIIO SEZ:
        /// "due to struct packing, we may get a corrupt header if we just dump the
        ///  struct to the file; to adress that, write every member individually"
        /// ... here we actually write out the header, member-by-member:
        if (!detail::write(&tga.idlen,          output) ||
            !detail::write(&tga.cmap_type,      output) ||
            !detail::write(&tga.type,           output) ||
            !detail::write(&tga.cmap_first,     output) ||
            !detail::write(&tga.cmap_length,    output) ||
            !detail::write(&tga.cmap_size,      output) ||
            !detail::write(&tga.x_origin,       output) ||
            !detail::write(&tga.y_origin,       output) ||
            !detail::write(&tga.width,          output) ||
            !detail::write(&tga.height,         output) ||
            !detail::write(&tga.bpp,            output) ||
            !detail::write(&tga.attr,           output))
        {
            imread_raise(CannotWriteError,
                "Failure writing TGA header");
        }
        
        if (targa_idlen) {
            if (!detail::write(targa_id.c_str(), output)) {
                imread_raise(CannotWriteError,
                    "Failure writing TGA identifier string");
            }
        }
    }
    
    
}
